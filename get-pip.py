#!/usr/bin/env python
"""
Bootstrap installation of pip into an existing Python installation or virtual environment.
"""
import os
import shutil
import sys
import tempfile
import urllib.request

def main():
    pip_url = "https://bootstrap.pypa.io/get-pip.py"
    script_path = os.path.join(tempfile.gettempdir(), "get-pip.py")

    print(f"Downloading {pip_url}...")
    urllib.request.urlretrieve(pip_url, script_path)
    print(f"Saved to {script_path}")

    print("Installing pip...")
    os.system(f'"{sys.executable}" {script_path}')

    print("Cleaning up...")
    os.remove(script_path)
    print("Pip installed successfully!")

if __name__ == "__main__":
    main()
