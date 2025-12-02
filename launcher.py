"""
Makerspace RAG - Unified Launcher
Single file to start everything: Ollama + Flask Web App
"""

import subprocess
import sys
import time
import socket
import webbrowser
import os
import platform
import signal

# Fix Windows console encoding
if platform.system() == 'Windows':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Configuration
OLLAMA_PORT = 11434
FLASK_PORT = 5000
REQUIRED_MODELS = ['llama3']
STARTUP_TIMEOUT = 60
CHECK_INTERVAL = 2


def print_banner():
    print("\n" + "=" * 60)
    print("  MAKERSPACE RAG - UNIFIED LAUNCHER")
    print("  Hogskolen i Ostfold")
    print("=" * 60)


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def check_ollama_running():
    try:
        import requests
        response = requests.get(f'http://127.0.0.1:{OLLAMA_PORT}/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False


def get_installed_models():
    try:
        import requests
        response = requests.get(f'http://127.0.0.1:{OLLAMA_PORT}/api/tags', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [m['name'].split(':')[0] for m in data.get('models', [])]
    except:
        pass
    return []


def start_ollama():
    print("\n[*] Starting Ollama server...")
    
    system = platform.system()
    
    if system == 'Windows':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        
        try:
            subprocess.Popen(
                ['ollama', 'serve'],
                startupinfo=startupinfo,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        except FileNotFoundError:
            print("  [X] Ollama not found! Install from: https://ollama.ai")
            return False
    else:
        try:
            subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except FileNotFoundError:
            print("  [X] Ollama not found! Install from: https://ollama.ai")
            return False
    
    print(f"  [.] Waiting for Ollama (max {STARTUP_TIMEOUT}s)...")
    start_time = time.time()
    
    while time.time() - start_time < STARTUP_TIMEOUT:
        if check_ollama_running():
            print("  [OK] Ollama is running!")
            return True
        time.sleep(CHECK_INTERVAL)
    
    print("  [X] Ollama failed to start")
    return False


def pull_model(model_name):
    print(f"  [.] Pulling model: {model_name}...")
    try:
        result = subprocess.run(['ollama', 'pull', model_name], capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"  [X] Error: {e}")
        return False


def ensure_models():
    print("\n[*] Checking required models...")
    
    installed = get_installed_models()
    print(f"  Installed: {installed if installed else 'none'}")
    
    missing = [m for m in REQUIRED_MODELS if m not in installed]
    
    if not missing:
        print("  [OK] All models installed!")
        return True
    
    print(f"  [!] Missing: {missing}")
    
    for model in missing:
        if not pull_model(model):
            print(f"  [X] Failed to pull {model}")
            return False
        print(f"  [OK] {model} installed!")
    
    return True


def run_flask_app():
    print("\n[*] Starting Flask web server...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        from app import app, load_vault
        
        print("  [.] Loading knowledge base...")
        load_vault()
        
        print(f"\n{'=' * 60}")
        print(f"  SERVER READY!")
        print(f"{'=' * 60}")
        print(f"\n  Chat:   http://localhost:{FLASK_PORT}/")
        print(f"  Admin:  http://localhost:{FLASK_PORT}/admin")
        print(f"  Login:  admin / makerspace2024")
        print(f"\n  Press Ctrl+C to stop")
        print(f"{'=' * 60}\n")
        
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f'http://localhost:{FLASK_PORT}/')
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        app.run(
            host='0.0.0.0',
            port=FLASK_PORT,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except ImportError as e:
        print(f"  [X] Import error: {e}")
        return False
    except Exception as e:
        print(f"  [X] Error: {e}")
        return False


def check_dependencies():
    print("\n[*] Checking dependencies...")
    
    required = ['flask', 'flask_login', 'sklearn', 'numpy', 'werkzeug', 'PyPDF2', 'ollama', 'requests']
    optional = ['pymupdf4llm', 'pdfplumber']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"  [X] Missing: {missing}")
        print("  Run: pip install -r requirements.txt")
        return False
    
    print("  [OK] All dependencies installed!")
    return True


def main():
    print_banner()
    
    if not check_dependencies():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    print("\n[*] Checking Ollama...")
    if check_ollama_running():
        print("  [OK] Ollama running!")
    else:
        print("  [!] Ollama not running")
        if not start_ollama():
            print("\n[X] Could not start Ollama!")
            input("\nPress Enter to exit...")
            sys.exit(1)
    
    if not ensure_models():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    try:
        run_flask_app()
    except KeyboardInterrupt:
        print("\n\nShutting down...")


if __name__ == '__main__':
    main()
