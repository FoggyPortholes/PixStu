import os
ROOT = os.path.dirname(os.path.dirname(__file__))
issues = []
for r,_,fs in os.walk(ROOT):
  for f in fs:
    if f.endswith('.py'):
      p = os.path.join(r,f)
      s = open(p, encoding='utf-8').read()
      if 'import app' in s or 'from app' in s:
        issues.append(p)
if os.path.exists(os.path.join(ROOT,'app')): issues.append('app/ directory present')
if os.path.exists(os.path.join(ROOT,'run_pcs.py')): issues.append('run_pcs.py present')
if issues:
  print('[FAIL] Migration incomplete:'); [print(' -',i) for i in issues]; raise SystemExit(1)
print('[OK] Migration clean')
