"""
get_fonts.py
------------
Downloads the TTF font files required by TranscribeFlow's PDF export.

Run once before starting the app:
    python get_fonts.py

NOTE: Japanese, Chinese, and Korean no longer need font files.
      They use ReportLab's built-in CID fonts (HeiseiKakuGo-W5,
      STSong-Light, HYGothic-Medium) which are part of the PDF
      standard and render correctly in all modern PDF viewers.

Only these three TTF files are needed:
    - NotoSans-Regular.ttf           (English / Latin)
    - NotoSansArabic-Regular.ttf     (Arabic, Persian, Urdu)
    - NotoSansDevanagari-Regular.ttf (Hindi)
"""

import requests
import os

FONTS_DIR = "fonts"
os.makedirs(FONTS_DIR, exist_ok=True)

fonts = {
    # English / Latin fallback
    "NotoSans-Regular.ttf":
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",

    # Arabic, Persian, Urdu
    "NotoSansArabic-Regular.ttf":
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansArabic/NotoSansArabic-Regular.ttf",

    # Hindi (Devanagari)
    "NotoSansDevanagari-Regular.ttf":
        "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
}

print("TranscribeFlow Font Downloader")
print("=" * 60)
print(f"Saving fonts to: ./{FONTS_DIR}/")
print()
print("NOTE: Japanese / Chinese / Korean PDFs use ReportLab's")
print("      built-in CID fonts — no download needed for CJK.")
print("=" * 60)

all_ok = True

for name, url in fonts.items():
    dest = os.path.join(FONTS_DIR, name)

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"✓ {name} already exists ({size_mb:.2f} MB), skipping.")
        continue

    print(f"⬇  Downloading {name} ...")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        with open(dest, "wb") as f:
            f.write(response.content)

        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"   ✓ Saved ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        all_ok = False

print("=" * 60)
if all_ok:
    print("All fonts ready. You can now start the app.")
else:
    print("Some fonts failed to download. Check your internet connection and retry.")
    print("The app will fall back to Helvetica for missing fonts,")
    print("which may show boxes for non-Latin characters.")
