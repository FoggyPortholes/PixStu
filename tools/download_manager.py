import os, requests, threading
from queue import Queue

class DownloadManager:
    def __init__(self, target_dir="loras"):
        self.queue = Queue()
        self.active = False
        self.target_dir = target_dir
        os.makedirs(target_dir, exist_ok=True)

    def add(self, url: str, filename: str, size_gb: float):
        self.queue.put((url, filename, size_gb))

    def _download_file(self, url, filename, size_gb):
        path = os.path.join(self.target_dir, filename)
        if os.path.exists(path):
            print(f"[SKIP] {filename} already exists")
            return path
        print(f"[DL] Starting {filename} (~{size_gb} GB)")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
                        if total % (50*1024*1024) < 8192:
                            print(f"[DL] {filename}: {total//1024//1024} MB")
        print(f"[OK] Download complete: {filename}")
        return path

    def worker(self):
        self.active = True
        while not self.queue.empty():
            url, filename, size_gb = self.queue.get()
            try:
                self._download_file(url, filename, size_gb)
            except Exception as e:
                print(f"[FAIL] Download {filename}: {e}")
            self.queue.task_done()
        self.active = False

    def run_async(self):
        if not self.active:
            threading.Thread(target=self.worker, daemon=True).start()
        else:
            print("[WAIT] Another download is already in progress.")
