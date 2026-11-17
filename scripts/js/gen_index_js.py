"""Generate isongraph-vector-js/src/index.js from isongraph-vector-ts/src/index.ts.

Run from anywhere: python scripts/js/gen_index_js.py   (needs npx + typescript
in isongraph-vector-ts). Pass --check to fail if the checked-in file differs.

tsc emits the JavaScript; this script swaps the ison-graph import and applies
the handful of JS-specific wording tweaks the checked-in file already carries.
"""
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
TS = ROOT / 'isongraph-vector-ts'
OUT = ROOT / 'isongraph-vector-js' / 'src' / 'index.js'
SP = tempfile.mkdtemp(prefix='igv-gen-')

subprocess.run(
    ['npx', 'tsc', 'src/index.ts', '--outDir', SP + '/gen', '--target', 'ES2020',
     '--module', 'ESNext', '--moduleResolution', 'node', '--skipLibCheck',
     '--newLine', 'lf', '--strict'],
    cwd=TS, check=True, shell=True)

s = open(SP + '/gen/index.js', encoding='utf8', newline='').read().replace('\r\n', '\n')
s = s.replace("from 'ison-graph-ts'", "from 'ison-graph-js'")

# header
start = s.index('/**')
end = s.index('*/', start) + 2
header = '''/**
 * ISONGraph Vector - Semantic Graph Extension for JavaScript
 *
 * Extends ISONGraph with embedding-based similarity search and semantic traversal.
 *
 * Generated from the TypeScript port's src/index.ts, so the two stay
 * behaviourally identical. Edit that file rather than this one.
 *
 * @author Mahesh Vaikri
 * @version 1.0.0
 */
'''
s = header + '\n' + s[end:].lstrip('\n')

def sub(old, new):
    global s
    assert old in s, old
    s = s.replace(old, new, 1)

sub('(Python/TypeScript/JavaScript/Rust/C++): a 32-bit FNV-1a',
    '(Python/TypeScript/JavaScript/Rust/C++/C#): a 32-bit FNV-1a')
sub('scoring a perfect 1.0 against each other. TypeScript escaped',
    'scoring a perfect 1.0 against each other. JavaScript escaped')
sub('a change to any of the embedFields re-embeds the node',
    'a change to any of embedFields re-embeds the node')
sub('addNode, nothing is re-embedded while no encoder is set.',
    'addNode, re-embedding needs an encoder; without one the vector is kept.')
sub('this._embedFields.some(f => f in properties)',
    'this._embedFields.some(f => f in (properties ?? {}))')

if '--check' in sys.argv:
    current = open(OUT, encoding='utf8', newline='').read().replace('\r\n', '\n')
    if current != s:
        sys.exit('%s is out of date: run python scripts/js/gen_index_js.py' % OUT)
    print('up to date:', OUT)
else:
    open(OUT, 'w', encoding='utf8', newline='').write(s)
    print('wrote', OUT, len(s.splitlines()), 'lines')
