"""Capture Hikari's live README media. Run with Python 3.13, Playwright, Edge and FFmpeg."""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw
from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8800"
SIZE = {"width": 1280, "height": 720}
DURATION = 11.4
EMOTES = ((2.0, "greet_wave", 2.9), (5.3, "starry_awe", 3.0), (8.6, "cat_smug", 2.6))
MEDIA = (
    "studio-hero.png", "studio-settings.png", "studio-memory.png",
    "hikari-demo.mp4", "hikari-demo.gif",
)


def port_listeners() -> int:
    result = subprocess.check_output([
        "powershell.exe", "-NoProfile", "-Command",
        "@(Get-NetTCPConnection -LocalPort 8800 -State Listen -ErrorAction SilentlyContinue).Count",
    ], text=True)
    return int(result.strip())


def prepare_studio(work: Path) -> Path:
    # server.py rebuilds the model and creates its own memory DB on startup.
    # Keep every write inside the temporary studio, including model/asset caches.
    studio = work / ROOT.name
    studio.mkdir()
    shutil.copy2(ROOT / "server.py", studio)
    for name in ("core", "modules", "web", "assets", "plugins", "engine"):
        shutil.copytree(ROOT / name, studio / name, ignore=shutil.ignore_patterns("__pycache__"))
    (studio / "tools").mkdir()
    shutil.copy2(ROOT / "tools" / "build_midori_deform_arms.py", studio / "tools")
    for name in ("models", "v28-inochi"):
        destination = (studio if name == "models" else work) / name
        destination.mkdir()
        source = (ROOT if name == "models" else ROOT.parent) / name
        for model in source.glob("Hikari-v28-BaseHD*.inp" if name == "models" else "Hikari-v28.inp"):
            shutil.copy2(model, destination)
    if not (work / "v28-inochi" / "Hikari-v28.inp").is_file():
        raise FileNotFoundError("Missing sibling v28-inochi/Hikari-v28.inp required by server.py")

    # Preserve the current Hikari look, with a fresh stage and no private history.
    settings_path = ROOT / "settings.json"
    current = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    settings = {
        **({"customizer": current["customizer"]} if "customizer" in current else {}),
        "ai": {"brainMode": "vercel_fish", "autoIdleEmotes": False},
        "rig": {"mouseLookAt": False},
        "stage": {"avatarScale": 0.96, "avatarY": 0.06},
        "camera": {"scale": 0.76, "x": 0.0, "y": 0.0},
    }
    (studio / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return studio


@contextmanager
def live_studio(studio: Path):
    if port_listeners():
        raise RuntimeError("Port 8800 is occupied. Stop the existing studio before capturing.")
    log = studio.parent / "server.log"
    with log.open("wb") as output:
        process = subprocess.Popen(
            [sys.executable, "server.py"], cwd=studio, stdout=output, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            deadline = time.monotonic() + 90
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    with urlopen(URL + "/", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.25)
            else:
                raise RuntimeError("Studio did not start:\n" + log.read_text(encoding="utf-8", errors="replace")[-4000:])
            yield
        finally:
            if process.poll() is None:
                try:
                    with urlopen(Request(URL + "/api/shutdown", method="POST"), timeout=5):
                        pass
                    process.wait(timeout=10)
                except (URLError, TimeoutError, subprocess.TimeoutExpired):
                    process.terminate()
                    process.wait(timeout=5)
            listeners = port_listeners()
            if listeners:
                raise RuntimeError(f"Port 8800 still has {listeners} listener(s)")
            print("Port 8800: 0 listeners", flush=True)


def settings_images(page: Page, output: Path) -> None:
    page.locator("#btn-settings >> visible=true").click()
    page.locator("[data-tab='voice'] >> visible=true").click()
    page.get_by_text("voices on your account", exact=False).filter(visible=True).wait_for()

    # Show a live model and the providers serving it.
    model = page.locator("select[aria-label='Vercel model'] >> visible=true")
    model.locator("optgroup").first.wait_for(state="attached")
    choices = model.locator("optgroup option").evaluate_all("opts => opts.map(o => o.value)")
    selected = "openai/gpt-5-mini" if "openai/gpt-5-mini" in choices else choices[0]
    model.select_option(selected)
    provider = page.locator("select[aria-label='Provider pinning'] >> visible=true")
    provider.locator("option[value]:not([value=''])").first.wait_for(state="attached")
    page.get_by_role("button", name="Create & Save a Voice").click()

    # Fit the relevant live controls in one screenshot of the scrollable tab.
    body = page.locator("#inspector-body")
    body.evaluate("""body => {
        const groups = new Set(['Vercel AI Gateway', 'Chat & Speech', 'Voice Designer (Fish Audio)']);
        for (const section of body.querySelectorAll(':scope > .s-group')) {
            const title = section.querySelector('h2').textContent;
            if (!groups.has(title)) section.hidden = true;
            if (title === 'Vercel AI Gateway' || title === 'Chat & Speech') {
                for (const row of section.querySelectorAll('.s-card > .s-row')) {
                    row.hidden = !row.querySelector('select[aria-label="Fish voice"], '
                        + 'select[aria-label="Vercel model"], select[aria-label="Provider pinning"]');
                }
            }
        }
        body.scrollTop = 0;
    }""")
    compact = page.add_style_tag(content="""
        #inspector-body [hidden] { display: none !important; }
        #inspector-body { zoom: .78; gap: 12px; }
        #inspector-body .s-group { gap: 8px; }
        #inspector-body .s-row { min-height: 48px; padding-block: 9px; gap: 10px; }
        #inspector-body .gw-picker > :not(select) { display: none; }
        #inspector-body .s-row:has(input[aria-label="Preview line"]) { display: none; }
        #inspector-body .s-more-body textarea { min-height: 0; height: 38px; }
        #inspector-body .s-ctrl:has(select[aria-label="Voice language"]) { max-width: none; gap: 6px; }
    """)
    page.mouse.move(5, 710)
    design_button = page.get_by_role("button", name="Design Voice", exact=True)
    expect(design_button).to_be_in_viewport(ratio=1)
    page.screenshot(path=output / "studio-settings.png")

    compact.evaluate("el => el.remove()")
    page.locator("[data-tab='memory'] >> visible=true").click()
    page.get_by_text("Persist conversation across restarts", exact=True).wait_for()
    page.add_style_tag(content="#inspector-body { zoom: .88; }")
    page.wait_for_timeout(650)
    page.screenshot(path=output / "studio-memory.png")


def record_stage(page: Page, output: Path) -> Path:
    """Record the live browser via CDP and preserve the timing of delivered frames."""
    stream = page.context.new_cdp_session(page)
    frames: list[tuple[float, Path]] = []

    def frame(event: dict) -> None:
        target = output / f"frame-{len(frames):04d}.jpg"
        target.write_bytes(base64.b64decode(event["data"]))
        frames.append((event["metadata"]["timestamp"], target))
        stream.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    stream.on("Page.screencastFrame", frame)
    stream.send("Page.startScreencast", {"format": "jpeg", "quality": 94, "maxWidth": 1280, "maxHeight": 720})
    start = time.monotonic()
    try:
        for offset, emote, duration in EMOTES:
            page.wait_for_timeout(max(0, (start + offset - time.monotonic()) * 1000))
            response = page.request.post(URL + "/api/emote", data={"emote": emote, "duration": duration})
            if not response.ok or not response.json().get("ok"):
                raise RuntimeError(f"Emote {emote} failed (HTTP {response.status})")
        page.wait_for_timeout(max(0, (start + DURATION - time.monotonic()) * 1000))
    finally:
        stream.send("Page.stopScreencast")
        stream.remove_listener("Page.screencastFrame", frame)
    if len(frames) < 100 or frames[-1][0] - frames[0][0] < 10:
        raise RuntimeError(f"Incomplete live recording ({len(frames)} frames)")

    timeline = output / "frames.ffconcat"
    entries = ["ffconcat version 1.0"]
    for (timestamp, file), (next_timestamp, _) in zip(frames, frames[1:]):
        entries.extend((f"file {file.name}", "option framerate 1000", f"duration {next_timestamp - timestamp:.6f}"))
    entries.extend((f"file {frames[-1][1].name}", "option framerate 1000"))
    timeline.write_text("\n".join(entries) + "\n", encoding="ascii")
    print(f"Recorded {len(frames)} live frames", flush=True)
    return timeline


def encode_demo(timeline: Path, ffmpeg: str) -> None:
    output = timeline.parent
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    mp4 = output / "hikari-demo.mp4"
    subprocess.run([
        *base, "-f", "concat", "-safe", "0", "-i", str(timeline), "-vf", "fps=30,scale=1280:720",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4),
    ], check=True)
    gif = output / "hikari-demo.gif"
    for fps, colors in ((18, 256), (15, 192), (12, 128)):
        subprocess.run([
            *base, "-i", str(mp4), "-filter_complex_threads", "1", "-filter_complex",
            f"fps={fps},scale=760:-2:flags=lanczos,split[v][p];[p]palettegen=max_colors={colors}[p];"
            "[v][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", str(gif),
        ], check=True)
        if gif.stat().st_size <= 8_000_000:
            break
    else:
        raise RuntimeError("GIF exceeds 8 MB")


def make_contact_sheet(gif: Path, destination: Path) -> None:
    with Image.open(gif) as image:
        thumb_size = (380, round(380 * image.height / image.width))
        tile_height = thumb_size[1] + 22
        sheet = Image.new("RGB", (4 * thumb_size[0], 2 * tile_height), "#1b1620")
        draw = ImageDraw.Draw(sheet)
        for index, seconds in enumerate((0.7, 1.7, 2.55, 3.4, 4.5, 6.25, 7.4, 9.65)):
            frame = round(seconds * (image.n_frames - 1) / DURATION)
            image.seek(frame)
            x, y = index % 4 * thumb_size[0], index // 4 * tile_height
            sheet.paste(image.convert("RGB").resize(thumb_size, Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 8, y + thumb_size[1] + 4), f"{seconds:.2f} s", fill="white")
        destination.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "media")
    parser.add_argument("--contact-sheet", type=Path, required=True)
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("ffmpeg must be on PATH")

    with TemporaryDirectory(prefix="hikari-readme-") as directory:
        work = Path(directory)
        studio = prepare_studio(work)
        with live_studio(studio), sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            try:
                page = browser.new_page(viewport=SIZE, device_scale_factor=1)
                page.goto(URL, wait_until="load")
                page.wait_for_function("document.querySelector('#hud-fps')?.textContent.match(/^\\d+/)")
                page.wait_for_timeout(800)
                page.screenshot(path=work / "studio-hero.png")
                settings_images(page, work)
                page.locator("button[aria-label='Close settings'] >> visible=true").click()
                page.mouse.move(5, 710)
                page.wait_for_timeout(550)
                timeline = record_stage(page, work)
            finally:
                browser.close()

        encode_demo(timeline, ffmpeg)
        make_contact_sheet(work / "hikari-demo.gif", args.contact_sheet)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name in MEDIA:
            shutil.copy2(work / name, args.output_dir / name)
            print(f"{name}: {(work / name).stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
