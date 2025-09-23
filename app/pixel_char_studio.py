import os
import socket
from chargen.studio import build_app


def _env_port() -> int | None:
    for key in ("PCS_PORT", "GRADIO_SERVER_PORT"):
        value = os.environ.get(key)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return None


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def _pick_port(host: str) -> int | None:
    env_port = _env_port()
    if env_port:
        return env_port
    default_port = 7860
    return default_port if _port_available(host, default_port) else None


if __name__ == "__main__":
    demo = build_app()
    server_name = os.environ.get("PCS_SERVER_NAME", "127.0.0.1")
    port = _pick_port(server_name)
    try:
        demo.launch(server_name=server_name, server_port=port)
    except OSError as exc:
        message = str(exc)
        if port and "Cannot find empty port" in message:
            demo.launch(server_name=server_name, server_port=None)
        else:
            raise
