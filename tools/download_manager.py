import os
import requests
import threading
from queue import Queue


class DownloadManager:
    """Single-queue downloader that avoids parallel transfers."""

    def __init__(self, target_dir: str = "loras") -> None:
        self.queue: "Queue[tuple[str, str, float]]" = Queue()
        self.active = False
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)

    def add(self, url: str, filename: str, size_gb: float) -> None:
        self.queue.put((url, filename, size_gb))

    def _download_file(self, url: str, filename: str, size_gb: float):
        path = os.path.join(self.target_dir, filename)
        if os.path.exists(path):
            print(f"[SKIP] {filename} already exists")
            return path
        print(f"[DL] Starting {filename} (~{size_gb} GB)")
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            written = 0
            with open(path, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
                        if written and written % (50 * 1024 * 1024) < 8192:
                            print(f"[DL] {filename}: {written // 1024 // 1024} MB")
        print(f"[OK] Download complete: {filename}")
        return path

    def _worker(self) -> None:
        self.active = True
        while not self.queue.empty():
            url, filename, size = self.queue.get()
            try:
                self._download_file(url, filename, size)
            except Exception as exc:  # pragma: no cover
                print(f"[FAIL] Download {filename}: {exc}")
            finally:
                self.queue.task_done()
        self.active = False

    def run_async(self) -> None:
        if self.active:
            print("[WAIT] Another download is already in progress.")
            return
        threading.Thread(target=self._worker, daemon=True).start()
