"""
Video-QA RAG System - Streamlit Application
Run this to start the Video Question Answering web interface
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit Video-QA application."""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the Video-QA Streamlit app
        app_path = os.path.join(script_dir, "video_qa", "app.py")
        
        # Run streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", app_path]
        print(f"Starting Video-QA Streamlit app from: {app_path}")
        subprocess.run(cmd, cwd=script_dir, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
