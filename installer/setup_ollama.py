"""
Makerspace RAG - Ollama Setup Script
Automatically downloads, installs, and configures Ollama with required models
"""

import os
import sys
import subprocess
import time
import urllib.request
import shutil
from pathlib import Path

# Required models
REQUIRED_MODELS = [
    'llama3',           # Main chat model
    'mxbai-embed-large' # Embedding model for semantic search
]

OLLAMA_DOWNLOAD_URL = 'https://ollama.com/download/OllamaSetup.exe'
OLLAMA_INSTALLER_PATH = Path('OllamaSetup.exe')


def print_status(message, status="INFO"):
    """Print formatted status message."""
    icons = {"INFO": "i", "OK": "+", "WARN": "!", "ERROR": "x", "WAIT": "~"}
    icon = icons.get(status, "*")
    print(f"  [{icon}] {message}")


def is_ollama_installed():
    """Check if Ollama is installed and accessible."""
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def is_ollama_running():
    """Check if Ollama service is running."""
    try:
        import requests
        response = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False


def download_ollama():
    """Download Ollama installer."""
    print_status("Downloading Ollama installer...", "WAIT")

    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  [⏳] Downloading: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(
            OLLAMA_DOWNLOAD_URL,
            OLLAMA_INSTALLER_PATH,
            reporthook=progress_hook
        )
        print()  # New line after progress
        print_status("Ollama installer downloaded", "OK")
        return True
    except Exception as e:
        print_status(f"Failed to download Ollama: {e}", "ERROR")
        return False


def install_ollama():
    """Install Ollama silently."""
    print_status("Installing Ollama (this may take a minute)...", "WAIT")

    try:
        # Run installer silently
        result = subprocess.run(
            [str(OLLAMA_INSTALLER_PATH), '/VERYSILENT', '/NORESTART'],
            capture_output=True,
            timeout=300  # 5 minute timeout
        )

        # Clean up installer
        if OLLAMA_INSTALLER_PATH.exists():
            OLLAMA_INSTALLER_PATH.unlink()

        # Wait for installation to complete
        time.sleep(5)

        if is_ollama_installed():
            print_status("Ollama installed successfully", "OK")
            return True
        else:
            print_status("Ollama installation may have failed", "WARN")
            return False

    except subprocess.TimeoutExpired:
        print_status("Installation timed out", "ERROR")
        return False
    except Exception as e:
        print_status(f"Installation failed: {e}", "ERROR")
        return False


def start_ollama_service():
    """Start Ollama service if not running."""
    if is_ollama_running():
        print_status("Ollama service already running", "OK")
        return True

    print_status("Starting Ollama service...", "WAIT")

    try:
        # Start ollama serve in background
        if sys.platform == 'win32':
            subprocess.Popen(
                ['ollama', 'serve'],
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # Wait for service to start
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if is_ollama_running():
                print_status("Ollama service started", "OK")
                return True

        print_status("Ollama service failed to start", "ERROR")
        return False

    except Exception as e:
        print_status(f"Failed to start Ollama: {e}", "ERROR")
        return False


def get_installed_models():
    """Get list of installed Ollama models."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    # Get base model name (without :tag)
                    base_name = model_name.split(':')[0]
                    models.append(base_name)
            return models
        return []
    except:
        return []


def pull_model(model_name):
    """Pull a specific Ollama model."""
    print_status(f"Downloading model: {model_name} (this may take a while)...", "WAIT")

    try:
        process = subprocess.Popen(
            ['ollama', 'pull', model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Stream output
        for line in process.stdout:
            line = line.strip()
            if line and ('pulling' in line.lower() or '%' in line):
                sys.stdout.write(f"\r  [⏳] {line[:60]:<60}")
                sys.stdout.flush()

        process.wait()
        print()  # New line

        if process.returncode == 0:
            print_status(f"Model {model_name} ready", "OK")
            return True
        else:
            print_status(f"Failed to pull {model_name}", "ERROR")
            return False

    except Exception as e:
        print_status(f"Error pulling {model_name}: {e}", "ERROR")
        return False


def ensure_models():
    """Ensure all required models are installed."""
    installed = get_installed_models()

    for model in REQUIRED_MODELS:
        if model in installed:
            print_status(f"Model {model} already installed", "OK")
        else:
            if not pull_model(model):
                return False

    return True


def setup_ollama():
    """Main setup function for Ollama."""
    print("\n" + "="*60)
    print("  Makerspace RAG - Ollama Setup")
    print("="*60 + "\n")

    # Step 1: Check/Install Ollama
    if is_ollama_installed():
        print_status("Ollama is installed", "OK")
    else:
        print_status("Ollama not found, installing...", "WARN")

        if not download_ollama():
            return False

        if not install_ollama():
            print_status("Please install Ollama manually from https://ollama.com", "ERROR")
            return False

    # Step 2: Start Ollama service
    if not start_ollama_service():
        return False

    # Step 3: Ensure models are installed
    if not ensure_models():
        return False

    print("\n" + "="*60)
    print_status("Ollama setup complete!", "OK")
    print("="*60 + "\n")

    return True


if __name__ == '__main__':
    success = setup_ollama()
    sys.exit(0 if success else 1)
