import hashlib
import re
import sys

PATS = [
    re.compile(r"/Users/[^/]+"),
    re.compile(r"/home/[^/]+"),
    re.compile(r"C:\\Users\\[^\\]+", re.IGNORECASE),
]
FNAME = re.compile(r"([\w\-]+)\.(png|jpe?g|json)", re.IGNORECASE)


def scrub(text: str) -> str:
    for pattern in PATS:
        text = pattern.sub("/home/<redacted>", text)
    text = FNAME.sub(
        lambda match: f"file_{hashlib.sha1(match.group(1).encode()).hexdigest()[:8]}.{match.group(2)}",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python tools/sanitize_reports.py <source> <destination>")
        raise SystemExit(1)
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8", errors="ignore") as handle:
        lines = [scrub(line) for line in handle]
    with open(dst, "w", encoding="utf-8") as handle:
        for line in lines:
            if line:
                handle.write(line + "\n")


if __name__ == "__main__":
    main()
