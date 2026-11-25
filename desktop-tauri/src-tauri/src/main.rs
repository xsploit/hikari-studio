//! Hikari Studio desktop shell.
//!
//! One Tauri window that shows a dark splash, makes sure the Python backend
//! (server.py, which itself launches hikari-live-engine.exe) is running, then
//! points the WebView at http://127.0.0.1:8800/. Every process we spawn goes
//! into a Windows Job Object with KILL_ON_JOB_CLOSE, so the backend and the
//! engine die with this app even if it crashes.
//!
//! On a normal quit we first ask a backend we started to stop by itself
//! (POST /api/shutdown, so it can end Fish calls and flush settings) and give
//! it 2.5 s before the hard kill.
//!
//! All waiting happens on Tokio tasks; the UI thread never blocks.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod job;

use std::{
    io::{self, Read as _, Write as _},
    net::{Ipv4Addr, SocketAddr, SocketAddrV4},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use tauri::{
    menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    window::Color,
    AppHandle, Manager, RunEvent, Theme, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder,
    WindowEvent,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    process::{Child, Command},
    sync::Notify,
    time::{sleep, timeout},
};

const HOST: &str = "127.0.0.1:8800";
const STAGE_URL: &str = "http://127.0.0.1:8800/";
const OBS_URL: &str = "http://127.0.0.1:8800/obs";
const MAIN: &str = "main";
const OBS: &str = "obs";
const BG: Color = Color(0x0e, 0x0e, 0x11, 0xff);
const MAX_RESTARTS: u32 = 3;
const BACKEND_ADDR: SocketAddr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 8800));
/// Bound on each step (connect, write, read) of the shutdown request.
const SHUTDOWN_IO_TIMEOUT: Duration = Duration::from_millis(400);
/// How long the backend gets to exit on its own after /api/shutdown.
const SHUTDOWN_GRACE: Duration = Duration::from_millis(5000);

/// Lets the frameless OBS window be dragged by holding the left mouse button.
const OBS_DRAG_JS: &str = r#"
window.addEventListener('mousedown', function (e) {
  if (e.button !== 0 || e.target.closest('input,button,select,textarea,a,[contenteditable]')) return;
  var t = window.__TAURI_INTERNALS__;
  if (t) { e.preventDefault(); t.invoke('plugin:window|start_dragging', { label: 'obs' }); }
}, true);
"#;

struct Shell {
    /// None if the Job Object could not be created (logged at startup).
    job: Option<job::Job>,
    /// Backends we started that are NOT in the job; tree-killed on exit.
    uncontained: Mutex<Vec<u32>>,
    /// Set once a quit starts; the supervisor never spawns or restarts after it.
    shutting_down: AtomicBool,
    /// True while the supervisor holds a live backend that we started.
    backend_owned: AtomicBool,
    /// Quit -> supervisor: stop the backend gracefully.
    stop: Notify,
    /// Supervisor -> quit: the backend is gone (or the grace period ended).
    stopped: Notify,
    /// Hard kill already done (it runs once).
    killed: AtomicBool,
    /// Cleanup finished; the next exit request is allowed through.
    exit_ready: AtomicBool,
    /// The splash page URL, so a restart can go back to it.
    splash: Mutex<Option<Url>>,
}

fn main() {
    let app = tauri::Builder::default()
        .manage(Shell {
            job: job::Job::new()
                .map_err(|e| warn(&format!("{e}. Falling back to taskkill /T on exit; a crash may leave the backend running.")))
                .ok(),
            uncontained: Mutex::new(Vec::new()),
            shutting_down: AtomicBool::new(false),
            backend_owned: AtomicBool::new(false),
            stop: Notify::new(),
            stopped: Notify::new(),
            killed: AtomicBool::new(false),
            exit_ready: AtomicBool::new(false),
            splash: Mutex::new(None),
        })
        .setup(|app| {
            let win = WebviewWindowBuilder::new(app, MAIN, WebviewUrl::App("index.html".into()))
                .title("Hikari Studio")
                .inner_size(1440.0, 900.0)
                .min_inner_size(800.0, 500.0)
                .center()
                .theme(Some(Theme::Dark))
                .background_color(BG)
                .build()?;
            *app.state::<Shell>().splash.lock().unwrap() = win.url().ok();
            build_tray(app.handle())?;
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(supervise(handle));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to start Hikari Studio shell");

    app.run(|app, event| match event {
        // Closing the main window quits the whole app (OBS window included).
        RunEvent::WindowEvent { label, event: WindowEvent::CloseRequested { .. }, .. }
            if label == MAIN =>
        {
            app.exit(0)
        }
        // Hold the exit open (without blocking this thread) until the backend
        // has had its chance to stop cleanly, then let it through.
        RunEvent::ExitRequested { api, .. } => {
            if !app.state::<Shell>().exit_ready.load(Ordering::SeqCst) {
                api.prevent_exit();
                begin_quit(app);
            }
        }
        // Last resort if exit happened some other way: never leave orphans.
        RunEvent::Exit => hard_kill(app),
        _ => {}
    });
}

/// Start the quit sequence once: hide windows, ask the supervisor for a
/// graceful stop, wait for it (bounded), hard-kill whatever is left, exit.
fn begin_quit(app: &AppHandle) {
    let shell = app.state::<Shell>();
    if shell.shutting_down.swap(true, Ordering::SeqCst) {
        return;
    }
    for w in app.webview_windows().values() {
        let _ = w.hide();
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        let shell = app.state::<Shell>();
        if shell.backend_owned.load(Ordering::SeqCst) {
            shell.stop.notify_one();
            // Request (<=1.2 s) + grace (2.5 s) + margin. Never hangs the quit.
            let limit = SHUTDOWN_IO_TIMEOUT * 3 + SHUTDOWN_GRACE + Duration::from_millis(800);
            if timeout(limit, shell.stopped.notified()).await.is_err() {
                warn("Backend supervisor did not report back in time; force-killing.");
            }
        }
        hard_kill(&app);
        shell.exit_ready.store(true, Ordering::SeqCst);
        app.exit(0);
    });
}

/// Kill everything we spawned, right now. Runs once.
fn hard_kill(app: &AppHandle) {
    let shell = app.state::<Shell>();
    shell.shutting_down.store(true, Ordering::SeqCst);
    if shell.killed.swap(true, Ordering::SeqCst) {
        return;
    }
    kill_job(&shell);
    // Fallback for backends outside the job: kill the whole tree by PID.
    for pid in shell.uncontained.lock().unwrap().drain(..) {
        let mut cmd = std::process::Command::new("taskkill");
        cmd.args(["/T", "/F", "/PID", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
        }
        if let Err(e) = cmd.status() {
            warn(&format!("taskkill /T /PID {pid} failed: {e}"));
        }
    }
}

fn kill_job(shell: &Shell) {
    if let Some(job) = &shell.job {
        if let Err(e) = job.kill_all() {
            warn(&e.to_string());
        }
    }
}

/// Problems the user can't see in a windowed app go to stderr and to
/// desktop-tauri/logs/desktop.log.
fn warn(msg: &str) {
    eprintln!("[hikari-desktop] {msg}");
    if cfg!(test) {
        return; // tests must not write into the real logs folder
    }
    if let Some(dir) = studio_dir() {
        let path = dir.join("desktop-tauri").join("logs").join("desktop.log");
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            use std::io::Write as _;
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            let _ = writeln!(f, "[unix {secs}] {msg}");
        }
    }
}

// ---------------------------------------------------------------- tray

fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    let toggle = MenuItem::with_id(app, "toggle", "Show / Hide Hikari", true, None::<&str>)?;
    let on_top = CheckMenuItem::with_id(app, "top", "Always on Top", true, false, None::<&str>)?;
    let obs = MenuItem::with_id(app, "obs", "Open OBS Transparent Window", true, None::<&str>)?;
    let reload = MenuItem::with_id(app, "reload", "Reload Stage", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let sep = PredefinedMenuItem::separator(app)?;
    let menu = Menu::with_items(app, &[&toggle, &on_top, &obs, &reload, &sep, &quit])?;

    let mut tray = TrayIconBuilder::with_id("hikari")
        .tooltip("Hikari Studio")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(move |app, e| match e.id.as_ref() {
            "toggle" => toggle_main(app),
            "top" => {
                if let (Some(w), Ok(on)) = (app.get_webview_window(MAIN), on_top.is_checked()) {
                    let _ = w.set_always_on_top(on);
                }
            }
            "obs" => open_obs(app.clone()),
            "reload" => reload_all(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, e| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = e
            {
                toggle_main(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

fn toggle_main(app: &AppHandle) {
    let Some(w) = app.get_webview_window(MAIN) else { return };
    if w.is_visible().unwrap_or(false) && !w.is_minimized().unwrap_or(false) {
        let _ = w.hide();
    } else {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

fn reload_all(app: &AppHandle) {
    if let Some(w) = app.get_webview_window(MAIN) {
        if on_stage(&w) {
            let _ = w.reload();
        } else {
            // Still on the splash: try the stage directly if the backend is up.
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                if backend_live().await {
                    show_stage(&app);
                }
            });
        }
    }
    if let Some(w) = app.get_webview_window(OBS) {
        let _ = w.reload();
    }
}

/// Frameless, transparent, always-on-top window showing /obs.
/// Built from an async task because creating a window inside a sync event
/// handler can deadlock WebView2 on Windows.
fn open_obs(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        if let Some(w) = app.get_webview_window(OBS) {
            let _ = w.unminimize();
            let _ = w.show();
            let _ = w.set_focus();
            return;
        }
        let url = WebviewUrl::External(OBS_URL.parse().expect("valid url"));
        let _ = WebviewWindowBuilder::new(&app, OBS, url)
            .title("Hikari OBS")
            .inner_size(480.0, 540.0)
            .decorations(false)
            .transparent(true)
            .shadow(false)
            .always_on_top(true)
            .background_color(Color(0, 0, 0, 0))
            .initialization_script(OBS_DRAG_JS)
            .build();
    });
}

// ---------------------------------------------------------------- main window state

fn on_stage(w: &WebviewWindow) -> bool {
    w.url().map(|u| u.as_str().starts_with(STAGE_URL)).unwrap_or(false)
}

fn show_stage(app: &AppHandle) {
    if let Some(w) = app.get_webview_window(MAIN) {
        if !on_stage(&w) {
            let _ = w.navigate(STAGE_URL.parse().expect("valid url"));
        }
    }
}

/// Update the splash text (no-op if the stage is showing).
fn status(app: &AppHandle, text: &str, is_error: bool) {
    let Some(w) = app.get_webview_window(MAIN) else { return };
    if on_stage(&w) {
        return;
    }
    let _ = w.eval(format!(
        "window.hikariStatus && hikariStatus({}, {})",
        js_string(text),
        is_error
    ));
}

/// Go back to the splash with a message (used when the backend dies).
fn show_splash(app: &AppHandle, text: &str) {
    let Some(w) = app.get_webview_window(MAIN) else { return };
    let splash = app.state::<Shell>().splash.lock().unwrap().clone();
    if let Some(mut url) = splash {
        url.set_fragment(Some(text));
        let _ = w.navigate(url);
    }
}

fn js_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            '<' => out.push_str("\\u003c"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------- backend supervision

/// One cheap HTTP/1.1 GET on /api/state; true if it answers 200.
async fn backend_live() -> bool {
    let probe = async {
        let mut s = TcpStream::connect(HOST).await.ok()?;
        s.write_all(b"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1:8800\r\nConnection: close\r\n\r\n")
            .await
            .ok()?;
        let mut buf = [0u8; 12];
        let mut n = 0;
        while n < buf.len() {
            let r = s.read(&mut buf[n..]).await.ok()?;
            if r == 0 {
                break;
            }
            n += r;
        }
        Some(n == 12 && buf.starts_with(b"HTTP/1.") && &buf[9..12] == b"200")
    };
    matches!(timeout(Duration::from_millis(1500), probe).await, Ok(Some(true)))
}

/// Sends one `POST /api/shutdown` to `addr`. Connect, write and read are
/// each bounded by `limit`, so this never hangs.
///
/// Ok: the request was delivered and the server answered 200, or closed /
/// stayed silent (it may already be exiting). Err: unreachable, timed out
/// before the request was sent, or answered with a non-200 status.
fn request_shutdown(addr: SocketAddr, limit: Duration) -> io::Result<()> {
    let mut s = std::net::TcpStream::connect_timeout(&addr, limit)?;
    s.set_write_timeout(Some(limit))?;
    s.set_read_timeout(Some(limit))?;
    s.write_all(
        format!("POST /api/shutdown HTTP/1.1\r\nHost: {addr}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .as_bytes(),
    )?;
    let mut buf = [0u8; 12];
    let mut n = 0;
    while n < buf.len() {
        match s.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(r) => n += r,
            // Silent (timeout) or reset while exiting: the request still arrived.
            Err(_) => break,
        }
    }
    if n == buf.len() && buf.starts_with(b"HTTP/1.") && &buf[9..12] != b"200" {
        let status = String::from_utf8_lossy(&buf[9..12]).into_owned();
        return Err(io::Error::other(format!("/api/shutdown answered HTTP {status}")));
    }
    Ok(())
}

/// Asks the backend to exit by itself, then polls `try_wait` every 50 ms for
/// up to `grace`. Returns true if it exited; false means the caller must
/// hard-kill (request failed, or it is still running).
async fn graceful_stop(child: &mut Child, addr: SocketAddr, grace: Duration) -> bool {
    if let Ok(Some(_)) = child.try_wait() {
        return true;
    }
    let sent = tokio::task::spawn_blocking(move || request_shutdown(addr, SHUTDOWN_IO_TIMEOUT)).await;
    match sent {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            warn(&format!("Graceful shutdown request failed ({e}); force-killing the backend."));
            return false;
        }
        Err(e) => {
            warn(&format!("Graceful shutdown task failed ({e}); force-killing the backend."));
            return false;
        }
    }
    let deadline = Instant::now() + grace;
    loop {
        if let Ok(Some(_)) = child.try_wait() {
            return true;
        }
        if Instant::now() >= deadline {
            warn(&format!(
                "server.py did not exit within {} ms of /api/shutdown; force-killing.",
                grace.as_millis()
            ));
            return false;
        }
        sleep(Duration::from_millis(50)).await;
    }
}

/// The backend exited on its own. Clear the flag *before* reading
/// shutting_down (begin_quit does the reverse), so a waiting quit is never
/// left hanging, and forget the PID so taskkill can never hit a reused one.
fn release_backend(app: &AppHandle, pid: Option<u32>) {
    let shell = app.state::<Shell>();
    shell.backend_owned.store(false, Ordering::SeqCst);
    if let Some(pid) = pid {
        shell.uncontained.lock().unwrap().retain(|p| *p != pid);
    }
    if shell.shutting_down.load(Ordering::SeqCst) {
        shell.stopped.notify_one();
    }
}

/// A quit arrived while we hold the backend: stop it gracefully, report back.
async fn stop_for_quit(app: &AppHandle, child: &mut Child, pid: Option<u32>) {
    let exited = graceful_stop(child, BACKEND_ADDR, SHUTDOWN_GRACE).await;
    let shell = app.state::<Shell>();
    shell.backend_owned.store(false, Ordering::SeqCst);
    if let (true, Some(pid)) = (exited, pid) {
        shell.uncontained.lock().unwrap().retain(|p| *p != pid);
    }
    shell.stopped.notify_one();
}

enum Wake {
    Exited(io::Result<std::process::ExitStatus>),
    Quit,
}

async fn supervise(app: AppHandle) {
    if backend_live().await {
        // Someone else already runs the backend; just use it, never kill it.
        show_stage(&app);
        return;
    }
    let Some(dir) = studio_dir() else {
        status(&app, "Could not find Hikari Studio backend (hikari-server.exe or server.py). Set HIKARI_STUDIO_DIR.", true);
        return;
    };
    let bundled_server = bundled_backend_exe(&dir);
    let python = if bundled_server.is_some() { PathBuf::new() } else { python_exe().await };
    let log = if dir.join("desktop-tauri").is_dir() {
        dir.join("desktop-tauri").join("logs").join("backend.log")
    } else {
        dir.join("logs").join("backend.log")
    };
    let mut restarts = 0u32;
    let shell = app.state::<Shell>();

    loop {
        if shell.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        let mut child = match spawn_backend(&app, bundled_server.as_deref(), &python, &dir, &log).await {
            Ok(c) => c,
            Err(e) => {
                let target = bundled_server.as_deref().unwrap_or(&python);
                status(&app, &format!("Could not start {}: {e}", target.display()), true);
                return;
            }
        };
        // Tokio forgets the PID once the process is reaped, so keep it now.
        let pid = child.id();

        // Wait until /api/state answers, or the process dies first.
        let started = Instant::now();
        let mut last_shown = 0u64;
        let ready = loop {
            if let Ok(Some(code)) = child.try_wait() {
                break Err(code);
            }
            if backend_live().await {
                break Ok(());
            }
            let secs = started.elapsed().as_secs();
            if secs != last_shown {
                last_shown = secs;
                status(&app, &format!("Starting Hikari Studio backend… {secs}s"), false);
            }
            let quit = tokio::select! {
                _ = sleep(Duration::from_millis(400)) => false,
                _ = shell.stop.notified() => true,
            };
            if quit {
                stop_for_quit(&app, &mut child, pid).await;
                return;
            }
        };
        if ready.is_ok() {
            show_stage(&app);
        }

        // Sleep (zero CPU) until the backend exits or a quit is requested.
        let code = match ready {
            Err(code) => code,
            Ok(()) => {
                let wake = tokio::select! {
                    r = child.wait() => Wake::Exited(r),
                    _ = shell.stop.notified() => Wake::Quit,
                };
                match wake {
                    Wake::Exited(Ok(code)) => code,
                    Wake::Exited(Err(_)) => {
                        release_backend(&app, pid);
                        return;
                    }
                    Wake::Quit => {
                        stop_for_quit(&app, &mut child, pid).await;
                        return;
                    }
                }
            }
        };
        release_backend(&app, pid);
        if shell.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        // Clear out an engine the dead backend may have left behind.
        kill_job(&shell);

        restarts += 1;
        if restarts > MAX_RESTARTS {
            show_splash(&app, "");
            sleep(Duration::from_millis(600)).await;
            status(
                &app,
                &format!("Backend stopped ({code}) and keeps failing. See {}", log.display()),
                true,
            );
            return;
        }
        show_splash(&app, &format!("Backend stopped ({code}). Restarting ({restarts}/{MAX_RESTARTS})…"));
        sleep(Duration::from_secs(2 * restarts as u64)).await;
        if backend_live().await {
            show_stage(&app);
            return;
        }
    }
}

async fn spawn_backend(app: &AppHandle, bundled: Option<&Path>, python: &Path, dir: &Path, log: &Path) -> std::io::Result<Child> {
    if let Some(parent) = log.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let out = tokio::fs::File::create(log).await?.into_std().await;
    let err = out.try_clone()?;

    let mut cmd = if let Some(exe) = bundled {
        let mut c = Command::new(exe);
        c.env("HIKARI_STUDIO_DIR", dir);
        c
    } else {
        let mut c = Command::new(python);
        c.arg("server.py");
        c
    };
    cmd.current_dir(dir)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);

    let child = cmd.spawn()?;
    app.state::<Shell>().backend_owned.store(true, Ordering::SeqCst);
    // Put it in the job right away: server.py only starts the engine after
    // its imports and model build, so the engine is always born inside it.
    let shell = app.state::<Shell>();
    let contained = match &shell.job {
        Some(job) => job
            .assign(&child)
            .map_err(|e| warn(&format!("{e}. Will taskkill /T the backend on exit instead.")))
            .is_ok(),
        None => false,
    };
    if !contained {
        if let Some(pid) = child.id() {
            shell.uncontained.lock().unwrap().push(pid);
        }
    }
    Ok(child)
}

/// HIKARI_STUDIO_DIR, else the first folder above the exe holding server.py,
/// else the folder this crate was built from.
fn bundled_backend_exe(dir: &Path) -> Option<PathBuf> {
    for rel in ["hikari-server/hikari-server.exe", "hikari-server.exe"] {
        let p = dir.join(rel);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn studio_dir() -> Option<PathBuf> {
    let has_backend = |p: &Path| bundled_backend_exe(p).is_some() || p.join("server.py").is_file();
    if let Some(d) = std::env::var_os("HIKARI_STUDIO_DIR").map(PathBuf::from) {
        if has_backend(&d) {
            return Some(d);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.ancestors().skip(1).find(|p| has_backend(p)) {
            return Some(d.to_path_buf());
        }
    }
    let built = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    has_backend(&built).then_some(built)
}

/// PYTHON env var, else the newest python.org install in
/// %LOCALAPPDATA%\Programs\Python, else `python` from PATH.
async fn python_exe() -> PathBuf {
    if let Some(p) = std::env::var_os("PYTHON").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        let root = Path::new(&local).join("Programs").join("Python");
        if let Ok(mut rd) = tokio::fs::read_dir(&root).await {
            let mut best: Option<((u32, u32), PathBuf)> = None;
            while let Ok(Some(e)) = rd.next_entry().await {
                let name = e.file_name().to_string_lossy().to_string();
                let Some(digits) = name.strip_prefix("Python") else { continue };
                let digits: String = digits.chars().take_while(char::is_ascii_digit).collect();
                if digits.len() < 2 {
                    continue;
                }
                let ver = (digits[..1].parse().unwrap_or(0), digits[1..].parse().unwrap_or(0));
                let exe = e.path().join("python.exe");
                if exe.is_file() && best.as_ref().map_or(true, |(b, _)| ver > *b) {
                    best = Some((ver, exe));
                }
            }
            if let Some((_, exe)) = best {
                return exe;
            }
        }
    }
    PathBuf::from("python")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// A localhost port with nothing listening on it.
    fn closed_port() -> SocketAddr {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap()
    }

    /// One-shot server that answers the first request with `reply` and
    /// returns what it received.
    fn answering_server(reply: &'static str) -> (SocketAddr, std::thread::JoinHandle<String>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let h = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 512];
            let n = s.read(&mut buf).unwrap();
            s.write_all(reply.as_bytes()).unwrap();
            String::from_utf8_lossy(&buf[..n]).into_owned()
        });
        (addr, h)
    }

    #[cfg(windows)]
    fn long_child() -> Child {
        tokio::process::Command::new("cmd")
            .args(["/c", "ping -n 30 127.0.0.1 >nul"])
            .creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW)
            .kill_on_drop(true)
            .spawn()
            .unwrap()
    }

    #[test]
    fn unreachable_port_is_a_fast_error() {
        let t = Instant::now();
        assert!(request_shutdown(closed_port(), SHUTDOWN_IO_TIMEOUT).is_err());
        assert!(t.elapsed() < Duration::from_secs(2), "took {:?}", t.elapsed());
    }

    #[test]
    fn silent_server_times_out_without_hanging() {
        // Accepts the TCP connection (OS backlog) but never answers.
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let t = Instant::now();
        assert!(request_shutdown(l.local_addr().unwrap(), SHUTDOWN_IO_TIMEOUT).is_ok());
        assert!(t.elapsed() < Duration::from_secs(2), "took {:?}", t.elapsed());
    }

    #[test]
    fn sends_post_and_accepts_200() {
        let (addr, h) = answering_server("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        request_shutdown(addr, SHUTDOWN_IO_TIMEOUT).unwrap();
        let req = h.join().unwrap();
        assert!(req.starts_with("POST /api/shutdown HTTP/1.1\r\n"), "{req}");
        assert!(req.contains("Content-Length: 0\r\n") && req.contains("Connection: close\r\n"));
    }

    #[test]
    fn non_200_is_an_error() {
        let (addr, h) = answering_server("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
        let e = request_shutdown(addr, SHUTDOWN_IO_TIMEOUT).unwrap_err();
        assert!(e.to_string().contains("404"), "{e}");
        h.join().unwrap();
    }

    /// Request fails -> give up at once (caller hard-kills), no 2.5 s wait.
    #[cfg(windows)]
    #[tokio::test]
    async fn graceful_stop_gives_up_fast_when_unreachable() {
        let mut child = long_child();
        let t = Instant::now();
        assert!(!graceful_stop(&mut child, closed_port(), SHUTDOWN_GRACE).await);
        assert!(t.elapsed() < Duration::from_secs(2), "took {:?}", t.elapsed());
        child.kill().await.unwrap();
    }

    /// Request delivered but the process keeps running -> false after the grace period.
    #[cfg(windows)]
    #[tokio::test]
    async fn graceful_stop_times_out_on_a_stuck_backend() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut child = long_child();
        let grace = Duration::from_millis(300);
        let t = Instant::now();
        assert!(!graceful_stop(&mut child, l.local_addr().unwrap(), grace).await);
        let took = t.elapsed();
        assert!(took >= grace && took < Duration::from_secs(3), "took {took:?}");
        child.kill().await.unwrap();
    }

    /// Request answered 200 and the process exits by itself -> true.
    #[cfg(windows)]
    #[tokio::test]
    async fn graceful_stop_sees_a_clean_exit() {
        let (addr, h) = answering_server("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        let mut child = tokio::process::Command::new("cmd")
            .args(["/c", "ping -n 2 127.0.0.1 >nul"]) // exits on its own after ~1 s
            .creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW)
            .spawn()
            .unwrap();
        assert!(graceful_stop(&mut child, addr, SHUTDOWN_GRACE).await);
        h.join().unwrap();
    }
}
