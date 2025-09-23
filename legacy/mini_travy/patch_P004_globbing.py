import re, sys


def main(src):
    code = open(src,'r',encoding='utf-8').read()
    # Ensure import
    if ' import glob' not in code:
        code = code.replace('import json, os, sys', 'import json, os, sys, glob')

    # Find pack handler
    pack_pat = re.compile(r'(elif\s+args\.cmd\s*==\s*[\'\"]pack[\'\"]\s*:\s*\n)(.*?)(\n\s*elif|\nif\s+__name__|$)', re.DOTALL)
    m = pack_pat.search(code)
    if not m: sys.exit("[P004] pack block not found.")
    head, body, tail = m.group(1), m.group(2), m.group(3)

    # After rows, cols = args.sheet, inject globbing
    rows_cols_pat = re.compile(r'(rows\s*,\s*cols\s*=\s*args\.sheet\s*\n)')
    if not rows_cols_pat.search(body): sys.exit("[P004] rows, cols assignment not found.")
    globber = (
        "        # P4: cross-platform globbing\n"
        "        frames = []\n"
        "        for pattern in args.frames:\n"
        "            matches = glob.glob(pattern)\n"
        "            if not matches and os.path.exists(pattern):\n"
        "                matches = [pattern]\n"
        "            frames.extend(sorted(matches))\n"
        "        if not frames:\n"
        "            frames = args.frames\n"
    )
    body = rows_cols_pat.sub(r'\\1' + globber, body, count=1)

    # Use frames for cols default
    body = re.sub(r'if\s+cols\s*==\s*0:\s*\n\s*cols\s*=\s*len\([^)]+\)', 'if cols == 0:\n            cols = len(frames)', body)
    # Replace loops args.frames -> frames
    body = re.sub(r'for\s+fp\s+in\s+args\.frames\s*:', 'for fp in frames:', body)

    out = code[:m.start()] + head + body + tail + code[m.end():]
    dst = src + ".P004.py"
    open(dst,'w',encoding='utf-8').write(out)
    print("Wrote", dst)


if __name__=='__main__':
    if len(sys.argv)!=2: sys.exit("Usage: python patch_P004_globbing.py spriterig_v1_1.py")
    main(sys.argv[1])
