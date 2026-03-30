#!/usr/bin/env python3
"""
BrainLink — one-command launcher.

Usage:
    python run.py              # install deps + start server + open browser
    python run.py --no-open    # start without opening browser
    python run.py --install    # only install dependencies
    python run.py --port 9000  # custom port
"""

import subprocess, sys, os, argparse, pathlib

HERE = pathlib.Path(__file__).resolve().parent
REQ  = HERE / "requirements.txt"
SRV  = HERE / "server.py"
VENV = HERE / ".venv"


def ensure_venv():
    """Create a venv if it doesn't exist."""
    if VENV.exists():
        return
    print("[1/3] Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
    print("      Done.")


def get_pip():
    """Return path to pip inside the venv."""
    if sys.platform == "win32":
        return str(VENV / "Scripts" / "pip.exe")
    return str(VENV / "bin" / "pip")


def get_python():
    """Return path to python inside the venv."""
    if sys.platform == "win32":
        return str(VENV / "Scripts" / "python.exe")
    return str(VENV / "bin" / "python")


def install_deps():
    """Install requirements into the venv."""
    pip = get_pip()
    print("[2/3] Installing dependencies...")
    subprocess.check_call(
        [pip, "install", "-r", str(REQ), "-q"],
        stdout=subprocess.DEVNULL
    )
    print("      Done.")


def run_server(port: int, no_open: bool):
    """Start the BrainLink server."""
    python = get_python()
    env = os.environ.copy()
    env["BRAINLINK_PORT"] = str(port)
    if no_open:
        env["BRAINLINK_NO_OPEN"] = "1"
    print(f"[3/3] Starting BrainLink on port {port}...")
    try:
        subprocess.check_call([python, str(SRV)], env=env)
    except KeyboardInterrupt:
        print("\n  BrainLink stopped. See you next time!")


def main():
    parser = argparse.ArgumentParser(description="BrainLink launcher")
    parser.add_argument("--port", type=int, default=8420, help="Server port (default: 8420)")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--install", action="store_true", help="Only install dependencies, don't start")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║     BrainLink — EEG Game Engine      ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    ensure_venv()
    install_deps()

    if args.install:
        print("\n  Dependencies installed. Run 'python run.py' to start.")
        return

    run_server(args.port, args.no_open)


if __name__ == "__main__":
    main()
