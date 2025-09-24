import hashlib
import sys


def simple_scrub_line(line: str) -> str:
    line = line.replace("/Users/", "/home/<redacted>/").replace("/home/", "/home/<redacted>/")
    tokens = line.split()
    cleaned = []
    for token in tokens:
        lower = token.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".json")):
            name = token.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            ext = token.rsplit(".", 1)[-1]
            digest = hashlib.sha1(name.encode()).hexdigest()[:8]
            cleaned.append(f"file_{digest}.{ext}")
        else:
            cleaned.append(token)
    return " ".join(cleaned)


def main(src: str, dst: str) -> None:
    with open(src, encoding="utf-8", errors="ignore") as handle:
        lines = [simple_scrub_line(line.rstrip()) for line in handle]
    with open(dst, "w", encoding="utf-8") as out:
        for line in lines:
            if line:
                out.write(line + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python tools/sanitize_reports.py <src> <dst>")
    main(sys.argv[1], sys.argv[2])
