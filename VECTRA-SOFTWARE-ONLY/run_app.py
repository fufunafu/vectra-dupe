"""Launch the Vectra-Dupe desktop app: start the FastAPI server in a background
thread, then open it in a native window.

    python run_app.py
"""
import socket, threading, time
import uvicorn
import webview

HOST, PORT = "127.0.0.1", 8731


def _serve():
    uvicorn.run("app.server:app", host=HOST, port=PORT, log_level="warning")


def _wait_until_up(timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((HOST, PORT), 0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


if __name__ == "__main__":
    threading.Thread(target=_serve, daemon=True).start()
    _wait_until_up()
    webview.create_window("Vectra-Dupe — 3D Face Reconstruction",
                          f"http://{HOST}:{PORT}/", width=1180, height=820,
                          min_size=(900, 640))
    webview.start()
