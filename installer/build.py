"""
Makerspace RAG - Build Script
Creates distributable installer package

Requirements:
- pip install pyinstaller
- Inno Setup 6.x installed (for Windows installer)

Usage:
    python installer/build.py
"""

import os
import sys
import shutil
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIST_DIR = PROJECT_ROOT / 'dist'
BUILD_DIR = PROJECT_ROOT / 'build'
INSTALLER_DIR = PROJECT_ROOT / 'installer'

OLLAMA_DOWNLOAD_URL = 'https://ollama.com/download/OllamaSetup.exe'


def print_step(step, message):
    print(f"\n{'='*60}")
    print(f"  Step {step}: {message}")
    print('='*60)


def clean_build():
    """Clean previous build artifacts."""
    print_step(1, "Cleaning previous builds")

    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            print(f"  Removing {dir_path}")
            shutil.rmtree(dir_path)

    print("  Clean complete")


def install_pyinstaller():
    """Ensure PyInstaller is installed."""
    print_step(2, "Checking PyInstaller")

    try:
        import PyInstaller
        print(f"  PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("  Installing PyInstaller...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyinstaller'],
                      check=True)
        print("  PyInstaller installed")


def build_executable():
    """Build executable with PyInstaller."""
    print_step(3, "Building executable with PyInstaller")

    spec_file = INSTALLER_DIR / 'makerspace_rag.spec'

    result = subprocess.run(
        [sys.executable, '-m', 'PyInstaller', str(spec_file), '--clean'],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print("  ERROR: PyInstaller build failed")
        return False

    # Check if output exists
    exe_path = DIST_DIR / 'MakerspaceRAG' / 'MakerspaceRAG.exe'
    if exe_path.exists():
        print(f"  Executable created: {exe_path}")
        return True
    else:
        print("  ERROR: Executable not found")
        return False


def download_ollama_installer():
    """Download Ollama installer for bundling."""
    print_step(4, "Downloading Ollama installer")

    ollama_path = INSTALLER_DIR / 'OllamaSetup.exe'

    if ollama_path.exists():
        print(f"  Ollama installer already exists: {ollama_path}")
        return True

    print(f"  Downloading from {OLLAMA_DOWNLOAD_URL}...")

    try:
        def progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(OLLAMA_DOWNLOAD_URL, ollama_path, reporthook=progress)
        print("\n  Download complete")
        return True
    except Exception as e:
        print(f"\n  ERROR: Failed to download Ollama: {e}")
        return False


def create_license():
    """Create LICENSE file if it doesn't exist."""
    license_path = PROJECT_ROOT / 'LICENSE'
    if not license_path.exists():
        license_path.write_text("""MIT License

Copyright (c) 2024 Hogskolen i Ostfold

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
        print("  Created LICENSE file")


def build_installer():
    """Build Windows installer with Inno Setup."""
    print_step(5, "Building Windows installer")

    # Check if Inno Setup is available
    iscc_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]

    iscc_path = None
    for path in iscc_paths:
        if os.path.exists(path):
            iscc_path = path
            break

    if not iscc_path:
        print("  WARNING: Inno Setup not found")
        print("  Download from: https://jrsoftware.org/isdl.php")
        print("  Skipping installer creation")
        return False

    create_license()

    iss_file = INSTALLER_DIR / 'setup.iss'
    result = subprocess.run([iscc_path, str(iss_file)], cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print("  Installer created successfully")
        return True
    else:
        print("  ERROR: Installer build failed")
        return False


def print_summary():
    """Print build summary."""
    print("\n" + "="*60)
    print("  BUILD COMPLETE")
    print("="*60)

    exe_path = DIST_DIR / 'MakerspaceRAG' / 'MakerspaceRAG.exe'
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n  Portable executable:")
        print(f"    {exe_path}")
        print(f"    Size: {size_mb:.1f} MB")

    # Find installer
    for installer in DIST_DIR.glob("MakerspaceRAG_Setup_*.exe"):
        size_mb = installer.stat().st_size / (1024 * 1024)
        print(f"\n  Windows installer:")
        print(f"    {installer}")
        print(f"    Size: {size_mb:.1f} MB")

    print("\n  To test the portable version:")
    print(f"    {exe_path}")
    print()


def main():
    print("\n" + "="*60)
    print("  MAKERSPACE RAG - BUILD SYSTEM")
    print("="*60)

    os.chdir(PROJECT_ROOT)

    # Step 1: Clean
    clean_build()

    # Step 2: Check PyInstaller
    install_pyinstaller()

    # Step 3: Build executable
    if not build_executable():
        print("\nBuild failed at executable stage")
        return 1

    # Step 4: Download Ollama
    download_ollama_installer()

    # Step 5: Build installer (optional)
    build_installer()

    # Summary
    print_summary()

    return 0


if __name__ == '__main__':
    sys.exit(main())
