#!/usr/bin/env python
"""
Launcher for Quantum Gravitational Wave Detector Web Interface

This script starts the web API server which also serves the frontend interface.
"""
import os
import sys
import subprocess
import webbrowser
import time
import signal
import platform

def get_localhost_url(port=8000):
    """Get localhost URL based on the port"""
    return f"http://localhost:{port}"

def open_browser(url):
    """Open web browser after a short delay"""
    def _open_browser():
        time.sleep(1.5)  # Wait for server to start
        print(f"Opening browser at {url}")
        webbrowser.open(url)
    
    import threading
    threading.Thread(target=_open_browser).start()

def handle_sigint(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    print("\nShutting down web interface...")
    sys.exit(0)

def main():
    """Start the web interface"""
    # Install required packages if needed
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
        
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Configure signal handler for graceful shutdown
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Print header
    print("\n" + "=" * 72)
    print(" Quantum Gravitational Wave Detector - Web Interface ".center(72, "="))
    print("=" * 72)
    
    # Check if frontend directory exists
    frontend_dir = os.path.join(script_dir, "frontend")
    if not os.path.exists(frontend_dir) or not os.path.exists(os.path.join(frontend_dir, "index.html")):
        print("Warning: Frontend files not found. The API will be available but not the web interface.")
    
    api_url = get_localhost_url()
    frontend_url = f"{api_url}/frontend/index.html"
    
    # Print access information
    print(f"\nServer starting...")
    print(f"API Documentation: {api_url}/docs")
    
    if os.path.exists(os.path.join(frontend_dir, "index.html")):
        print(f"Web Interface: {frontend_url}")
        
        # Open browser automatically (unless running in WSL or other limited environment)
        if not (platform.system() == "Linux" and "microsoft" in platform.uname().release.lower()):
            open_browser(api_url)
    
    print("\nPress Ctrl+C to stop the server\n")
    
    # Import and run the web_api server
    from qgw_detector.web_api import start_server
    start_server()

if __name__ == "__main__":
    main()
