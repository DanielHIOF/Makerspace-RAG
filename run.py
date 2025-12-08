"""
Makerspace RAG - Application Entry Point
Run this file to start the Flask server with React frontend
"""

import os
import sys

# Check if React build exists
REACT_BUILD = os.path.join(os.path.dirname(__file__), 'app', 'static', 'react', 'index.html')
REACT_EXISTS = os.path.exists(REACT_BUILD)

from app import create_app

# Create the application
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'

    print(f"\n{'='*60}")
    print("  Makerspace RAG - Starting Server")
    print(f"{'='*60}")

    if REACT_EXISTS:
        print(f"  Frontend: React (built)")
    else:
        print(f"  Frontend: NOT BUILT - run 'npm run build' in frontend/")
        print(f"            Or use 'npm run dev' for development")

    print(f"\n  URL: http://localhost:{port}")
    print(f"  Debug: {debug}")
    print(f"{'='*60}")
    print("\n  For development with hot-reload:")
    print("    1. Run this server (python run.py)")
    print("    2. In another terminal: cd frontend && npm run dev")
    print("    3. Open http://localhost:3000")
    print(f"\n  For production (built React):")
    print("    1. cd frontend && npm run build")
    print(f"    2. Open http://localhost:{port}")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=debug)
