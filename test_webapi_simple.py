#!/usr/bin/env python
"""
Simple test script for the web API
"""
import subprocess
import time
import sys

def main():
    # Start the web API server
    print("Starting web API server...")
    server_process = subprocess.Popen(["python", "-m", "qgw_detector.web_api"])
    
    print("Server started. Press Ctrl+C to exit.")
    try:
        # Keep the server running until user interrupts
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server_process.terminate()
        server_process.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()
