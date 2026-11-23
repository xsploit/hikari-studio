//! Hikari Studio desktop shell.
//!
//! One Tauri window that shows a dark splash, makes sure the Python backend
//! (server.py, which itself launches hikari-live-engine.exe) is running, then
//! points the WebView at http://127.0.0.1:8800/. Every process we spawn goes
//! into a Windows Job Object with KILL_ON_JOB_CLOSE, so the backend and the
//! engine die with this app even if it crashes.
//!
//! All waiting happens on Tokio tasks; the UI thread never blocks.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod job;

use std::{
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
    time::{sleep, timeout},
};

const HOST: &str = "127.0.0.1:8800";
const STAGE_URL: &str = "http://127.0.0.1:8800/";
const OBS_URL: &str = "http://127.0.0.1:8800/obs";
const MAIN: &str = "main";
const OBS: &str = "obs";
const BG: Color = Color(0x0e, 0x0e, 0x11, 0xff);
const MAX_RESTARTS: u32 = 3;

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
    shutting_down: AtomicBool,
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
        RunEvent::ExitRequested { .. } | RunEvent::Exit => shutdown(app),
        _ => {}
    });
}

/// Kill everything we spawned. Safe to call more than once.
fn shutdown(app: &AppHandle) {
    let shell = app.state::<Shell>();
    if shell.shutting_down.swap(true, Ordering::SeqCst) {
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

async fn supervise(app: AppHandle) {
    if backend_live().await {
        // Someone else already runs the backend; just use it, never kill it.
        show_stage(&app);
        return;
    }
    let Some(dir) = studio_dir() else {
        status(&app, "Could not find server.py. Set HIKARI_STUDIO_DIR to the hikari-studio folder.", true);
        return;
    };
    let python = python_exe().await;
    let log = dir.join("desktop-tauri").join("logs").join("backend.log");
    let mut restarts = 0u32;

    loop {
        let mut child = match spawn_backend(&app, &python, &dir, &log).await {
            Ok(c) => c,
            Err(e) => {
                status(&app, &format!("Could not start {}: {e}", python.display()), true);
                return;
            }
        };

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
            sleep(Duration::from_millis(400)).await;
        };
        if ready.is_ok() {
            show_stage(&app);
        }

        // Sleep (zero CPU) until the backend exits.
        let code = match ready {
            Err(code) => code,
            Ok(()) => match child.wait().await {
                Ok(code) => code,
                Err(_) => return,
            },
        };
        if app.state::<Shell>().shutting_down.load(Ordering::SeqCst) {
            return;
        }
        // Clear out an engine the dead backend may have left behind.
        let shell = app.state::<Shell>();
        kill_job(&shell);
        if let Some(pid) = child.id() {
            shell.uncontained.lock().unwrap().retain(|p| *p != pid);
        }

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

async fn spawn_backend(app: &AppHandle, python: &Path, dir: &Path, log: &Path) -> std::io::Result<Child> {
    if let Some(parent) = log.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let out = tokio::fs::File::create(log).await?.into_std().await;
    let err = out.try_clone()?;

    let mut cmd = Command::new(python);
    cmd.arg("server.py")
        .current_dir(dir)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);

    let child = cmd.spawn()?;
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
fn studio_dir() -> Option<PathBuf> {
    let has_server = |p: &Path| p.join("server.py").is_file();
    if let Some(d) = std::env::var_os("HIKARI_STUDIO_DIR").map(PathBuf::from) {
        if has_server(&d) {
            return Some(d);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.ancestors().skip(1).find(|p| has_server(p)) {
            return Some(d.to_path_buf());
        }
    }
    let built = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    has_server(&built).then_some(built)
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
