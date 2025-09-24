import sys, hashlib

# Minimal PII scrubber without regex backslash escapes
# - Redacts common UNIX-style home paths
# - Hashes filenames found in simple tokens ending with .png/.jpg/.jpeg/.json

def simple_scrub_line(s):
    # redact unix home paths
    s = s.replace("/Users/", "/home/<redacted>/").replace("/home/", "/home/<redacted>/")
    # collapse whitespace
    s = " ".join(s.split())
    # filename hashing (very simple heuristic)
    out = []
    for tok in s.split(" "):
        low = tok.lower()
        if low.endswith(".png") or low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".json"):
            name = tok.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            ext = tok.rsplit(".", 1)[-1]
            hashed = "file_" + hashlib.sha1(name.encode()).hexdigest()[:8] + "." + ext
            out.append(hashed)
        else:
            out.append(tok)
    return " ".join(out)

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8", errors="ignore") as f:
        lines = [simple_scrub_line(x) for x in f]
    with open(dst, "w", encoding="utf-8") as g:
        for ln in lines:
            if ln: g.write(ln+"\n")
