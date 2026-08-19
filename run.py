"""
Zero-Configuration Auto-Browser Launcher for NeuronFrontier-LM Dashboard.
Starts the FastAPI server and opens http://localhost:8000 in your default web browser.
"""

import sys
import time
import webbrowser
import threading
import uvicorn

def open_browser():
    time.sleep(1.2)
    url = "http://localhost:8000"
    print(f"\n🚀 Launching NeuronFrontier-LM Console at {url} ...\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

def main():
    print("=" * 70)
    print(" 🚀 STARTING AWS TRAINIUM FRONTIER SPEEDRUN CONSOLE ")
    print("=" * 70)
    print("Features:")
    print(" • Live 30-Minute Speedrun Monitor with Watchdog Timer")
    print(" • Validation BPB (bits/byte) & MFU Real-Time Telemetry")
    print(" • Custom NKI Hardware Kernel Benchmarking")
    print(" • 1-Click Automated Judge Tour for Competition Reviewers")
    print("=" * 70)
    
    # Launch browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Uvicorn
    uvicorn.run("neuron_frontier.web.app:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
