#!/usr/bin/env python3
"""
Daily WL QA — one-step installer.

Run once after cloning:
    python3 install.py          (Linux / macOS)
    python  install.py          (Windows)

What it does:
  1. Checks Python version (3.10+ required).
  2. Installs Python dependencies via pip.
  3. Generates the app icon (icon.png / icon.ico).
  4. Linux: creates a .desktop launcher and installs hicolor icons so the
     app appears in your application menu.
  5. Windows: creates a shortcut on the Desktop (uses PowerShell — no
     extra packages needed).
"""

import sys
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"


# ── 1. Python version check ───────────────────────────────────────────────────

if sys.version_info < (3, 10):
    print(f"ERROR: Python 3.10 or newer is required (you have {sys.version}).")
    sys.exit(1)

print(f"Python {sys.version.split()[0]}  ✓")


# ── 2. Install dependencies ───────────────────────────────────────────────────

print("\nInstalling dependencies …")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", str(HERE / "requirements.txt"),
     "--quiet", "--disable-pip-version-check"],
    capture_output=False,
)
if result.returncode != 0:
    print("\nERROR: pip install failed. Check the output above.")
    sys.exit(1)
print("Dependencies installed  ✓")


# ── 3. Generate icon ──────────────────────────────────────────────────────────

def _make_icon():
    import math
    from PIL import Image, ImageDraw, ImageFont

    FONT_CANDIDATES = [
        # Linux
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    FONT_PATH = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)

    BG         = (17,  17,  17)
    FIELD_DARK = (28,  28,  38)
    FIELD_MID  = (45,  45,  60)
    BORDER_COL = (80, 160, 255)
    CROSS_COL  = (80, 160, 255, 200)
    VOID_COL   = (255, 183,  77)
    VOID_RING  = (255, 183,  77, 180)
    MEC_COL    = (255, 255, 255, 160)
    PASS_COL   = (102, 187, 106)

    def draw_icon(size):
        s    = size
        img  = Image.new("RGBA", (s, s), BG + (255,))
        draw = ImageDraw.Draw(img, "RGBA")
        pad  = max(1, s // 16)
        draw.ellipse([pad, pad, s - pad, s - pad], fill=FIELD_DARK + (255,))
        margin = s * 0.18
        fx0, fy0, fx1, fy1 = margin, margin, s - margin, s - margin
        draw.rectangle([fx0, fy0, fx1, fy1], fill=FIELD_MID + (255,))
        bw = max(1, s // 64)
        draw.rectangle([fx0, fy0, fx1, fy1], outline=BORDER_COL + (220,), width=bw)
        cx, cy = s / 2, s / 2
        gap, cw = s * 0.08, max(1, s // 80)
        draw.line([(fx0 + 2, cy), (cx - gap, cy)], fill=CROSS_COL, width=cw)
        draw.line([(cx + gap, cy), (fx1 - 2, cy)], fill=CROSS_COL, width=cw)
        draw.line([(cx, fy0 + 2), (cx, cy - gap)], fill=CROSS_COL, width=cw)
        draw.line([(cx, cy + gap), (cx, fy1 - 2)], fill=CROSS_COL, width=cw)
        wc_r = s * 0.16
        draw.ellipse([cx - wc_r, cy - wc_r, cx + wc_r, cy + wc_r],
                     outline=MEC_COL, width=max(1, s // 96))
        off  = s * 0.08
        vr   = max(2, s // 22)
        vx, vy = cx + off * 0.7, cy - off * 0.5
        draw.ellipse([vx - vr, vy - vr, vx + vr, vy + vr], fill=VOID_COL + (255,))
        draw.ellipse([vx - vr * 1.7, vy - vr * 1.7, vx + vr * 1.7, vy + vr * 1.7],
                     outline=VOID_RING, width=max(1, s // 96))
        if s >= 48:
            draw.line([(cx, cy), (vx, vy)], fill=VOID_COL + (200,),
                      width=max(1, s // 96))
        if s >= 64:
            fs = max(8, s // 8)
            try:
                font = ImageFont.truetype(FONT_PATH, fs) if FONT_PATH else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            label = "WL"
            bbox  = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw / 2, fy1 - th - s * 0.07), label,
                      font=font, fill=PASS_COL + (230,))
        if s >= 32:
            dr = max(3, s // 18)
            dx, dy = s - pad - dr - 2, pad + dr + 2
            draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr],
                         fill=PASS_COL + (255,))
        return img

    master = draw_icon(512)
    master.save(str(HERE / "icon_512.png"))
    draw_icon(256).save(str(HERE / "icon.png"))
    ico_imgs = [draw_icon(s) for s in [16, 32, 48, 64, 128, 256]]
    ico_imgs[0].save(
        str(HERE / "icon.ico"),
        format="ICO",
        sizes=[(s, s) for s in [16, 32, 48, 64, 128, 256]],
        append_images=ico_imgs[1:],
    )


print("\nGenerating icon …")
try:
    _make_icon()
    print("Icon generated  ✓")
except Exception as exc:
    print(f"WARNING: icon generation failed ({exc}) — the app will still run without it.")


# ── 4. Platform-specific installation ─────────────────────────────────────────

PLATFORM = sys.platform   # "linux", "darwin", "win32"

# ── Linux ────────────────────────────────────────────────────────────────────
if PLATFORM.startswith("linux"):
    from PIL import Image as PILImage

    # hicolor icons
    for size in [16, 32, 48, 64, 128, 256, 512]:
        icon_dir = Path.home() / ".local/share/icons/hicolor" / f"{size}x{size}" / "apps"
        icon_dir.mkdir(parents=True, exist_ok=True)
        src = HERE / "icon_512.png"
        if src.exists():
            PILImage.open(src).resize((size, size), PILImage.LANCZOS).save(
                str(icon_dir / "daily-wl-qa.png")
            )

    # .desktop file
    desktop_dir = Path.home() / ".local/share/applications"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    desktop_file = desktop_dir / "daily-wl-qa.desktop"
    desktop_file.write_text(
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        "Name=Daily WL QA\n"
        "GenericName=Winston-Lutz QA Tool\n"
        "Comment=Daily Winston-Lutz mechanical isocenter QA — Elekta Versa HD / MIMI Phantom\n"
        f'Exec={sys.executable} "{HERE / "wl_qa_tool.py"}"\n'
        "Icon=daily-wl-qa\n"
        "Terminal=false\n"
        "Categories=Science;MedicalSoftware;\n"
        "Keywords=Winston-Lutz;QA;LINAC;isocenter;MIMI;\n"
        "StartupWMClass=wl_qa_tool\n"
    )
    desktop_file.chmod(0o755)

    # Refresh caches (non-fatal if tools unavailable)
    subprocess.run(["update-desktop-database", str(desktop_dir)],
                   capture_output=True)
    subprocess.run(["gtk-update-icon-cache", "-f", "-t",
                    str(Path.home() / ".local/share/icons/hicolor")],
                   capture_output=True)

    print("Linux desktop entry installed  ✓")
    print(f"  → {desktop_file}")

# ── Windows ──────────────────────────────────────────────────────────────────
elif PLATFORM == "win32":
    desktop = Path.home() / "Desktop"
    target  = str(HERE / "wl_qa_tool.py")
    icon    = str(HERE / "icon.ico")
    lnk     = str(desktop / "Daily WL QA.lnk")

    ps_script = f"""
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut('{lnk}')
$sc.TargetPath  = '{sys.executable}'
$sc.Arguments   = '"{target}"'
$sc.IconLocation = '{icon}'
$sc.WorkingDirectory = '{HERE}'
$sc.Description = 'Daily Winston-Lutz QA Tool'
$sc.Save()
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Windows Desktop shortcut created  ✓")
        print(f"  → {lnk}")
    else:
        print("WARNING: could not create Desktop shortcut automatically.")
        print(f"  To run: double-click  {HERE / 'run_wl_qa.bat'}")

# ── macOS ────────────────────────────────────────────────────────────────────
elif PLATFORM == "darwin":
    print("macOS: no launcher created automatically.")
    print(f"  To run:  python3 \"{HERE / 'wl_qa_tool.py'}\"")


# ── Done ──────────────────────────────────────────────────────────────────────

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Installation complete!

  To launch the app:
    python3 "{HERE / 'wl_qa_tool.py'}"

  Or search "Daily WL QA" in your application menu.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
