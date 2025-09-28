"""
Simple Comment Cleaner (lightweight replacement for Comment Cleaner Pro)
- Creates .bak backups by default
- Removes comments from: .py, .js, .ts, .jsx, .tsx, .java, .c, .cpp, .h, .html, .css, .scss, .go, .rs, .php, .rb, .swift, .kt, .kts, .sh, .ps1, .ipynb

Usage:
  python scripts/ccp.py "." --recursive
  python scripts/ccp.py path --no-backup --recursive

WARNING: This modifies files in-place. Backups are created with .bak extension by default.
"""

import argparse
import io
import json
import os
import shutil
import sys
import tokenize

EXTENSIONS = [
    '.py','.js','.ts','.jsx','.tsx','.java','.c','.cpp','.h','.html','.htm',
    '.css','.scss','.go','.rs','.php','.rb','.swift','.kt','.kts','.sh','.ps1', '.ipynb'
]


def remove_comments_python(src: str) -> str:
    """Remove Python comments using the tokenize module (keeps code and strings intact)."""
    try:
        src_bytes = src.encode('utf-8')
        out_tokens = []
        g = tokenize.tokenize(io.BytesIO(src_bytes).readline)
        for toknum, tokval, start, end, line in g:
            if toknum == tokenize.COMMENT:
                continue
            if toknum == tokenize.ENCODING:
                continue
            out_tokens.append((toknum, tokval))
        return tokenize.untokenize(out_tokens).decode('utf-8')
    except Exception:
        return '\n'.join([l for l in src.splitlines() if not l.strip().startswith('#')]) + '\n'


def remove_comments_cstyle(src: str) -> str:
    """Remove // and /* */ comments while attempting to preserve string literals."""
    out = []
    i = 0
    n = len(src)
    in_single = False
    in_double = False
    in_backtick = False
    in_block = False
    in_line = False
    esc = False
    while i < n:
        ch = src[i]
        nxt = src[i+1] if i+1 < n else ''
        if in_block:
            if ch == '*' and nxt == '/':
                in_block = False
                i += 2
                continue
            else:
                i += 1
                continue
        if in_line:
            if ch == '\n':
                in_line = False
                out.append(ch)
            i += 1
            continue
        if not (in_single or in_double or in_backtick):
            if ch == '/' and nxt == '*':
                in_block = True
                i += 2
                continue
            if ch == '/' and nxt == '/':
                in_line = True
                i += 2
                continue
        if ch == '\\' and (in_single or in_double or in_backtick):
            out.append(ch)
            if i+1 < n:
                out.append(src[i+1])
                i += 2
                continue
        if not in_single and not in_double and ch == '"':
            in_double = True
            out.append(ch)
            i += 1
            continue
        if in_double and ch == '"':
            in_double = False
            out.append(ch)
            i += 1
            continue
        if not in_single and not in_double and ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if in_single and ch == "'":
            in_single = False
            out.append(ch)
            i += 1
            continue
        if not in_single and not in_double and ch == '`':
            in_backtick = not in_backtick
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def remove_comments_shell(src: str) -> str:
    lines = src.splitlines()
    out = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith('#'):
            if i == 0 and stripped.startswith('#!'):
                out.append(line)
            else:
                continue
        else:
            new_line = ''
            in_s = False
            in_d = False
            esc = False
            for ch in line:
                if ch == '\\' and not esc:
                    esc = True
                    new_line += ch
                    continue
                if ch == '"' and not esc and not in_s:
                    in_d = not in_d
                if ch == "'" and not esc and not in_d:
                    in_s = not in_s
                if ch == '#' and not in_s and not in_d:
                    break
                new_line += ch
                esc = False
            out.append(new_line.rstrip())
    return '\n'.join(out) + '\n'


def process_file(path, no_backup=False, verbose=False):
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except Exception as e:
        if verbose:
            print(f"Skipping binary or unreadable file: {path} ({e})")
        return
    try:
        src = raw.decode('utf-8')
    except UnicodeDecodeError:
        if verbose:
            print(f"Skipping non-utf8 file: {path}")
        return

    if not no_backup:
        bak = path + '.bak'
        shutil.copy2(path, bak)

    new_src = None
    if ext == '.py':
        new_src = remove_comments_python(src)
    elif ext in ['.sh']:
        new_src = remove_comments_shell(src)
    elif ext == '.ipynb':
        try:
            nb = json.loads(src)
            changed = False
            for cell in nb.get('cells', []):
                if cell.get('cell_type') == 'code':
                    source = cell.get('source', [])
                    if isinstance(source, list):
                        src_code = ''.join(source)
                    else:
                        src_code = source
                    cleaned = remove_comments_python(src_code)
                    if cleaned != src_code:
                        cell['source'] = [ln+('\n' if not ln.endswith('\n') else '') for ln in cleaned.splitlines()]
                        changed = True
            if not changed:
                if verbose:
                    print(f"No changes in notebook: {path}")
                return
            new_src = json.dumps(nb, ensure_ascii=False, indent=1)
        except Exception as e:
            if verbose:
                print(f"Failed to parse notebook {path}: {e}")
            return
    elif ext in ['.js','.ts','.jsx','.tsx','.java','.c','.cpp','.h','.html','.htm','.css','.scss','.go','.rs','.php','.rb','.swift','.kt','.kts']:
        new_src = remove_comments_cstyle(src)
    else:
        if verbose:
            print(f"Skipping unsupported extension: {path}")
        return

    if new_src is None:
        return

    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_src)
    if verbose:
        print(f"Cleaned comments: {path}")


def walk_and_process(root, recursive=True, no_backup=False, verbose=False):
    for dirpath, dirnames, filenames in os.walk(root):
        parts = dirpath.split(os.sep)
        if '.git' in parts or '__pycache__' in parts:
            continue
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            _, ext = os.path.splitext(fn)
            if ext.lower() in EXTENSIONS:
                process_file(path, no_backup=no_backup, verbose=verbose)
        if not recursive:
            break


def main():
    parser = argparse.ArgumentParser(description='Comment Cleaner (lightweight)')
    parser.add_argument('path', help='Path to project root or file', nargs='?', default='.')
    parser.add_argument('--recursive', action='store_true', help='Recurse into subdirectories')
    parser.add_argument('--no-backup', action='store_true', help="Don't create .bak backups")
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    root = os.path.abspath(args.path)
    if os.path.isfile(root):
        process_file(root, no_backup=args.no_backup, verbose=args.verbose)
    else:
        walk_and_process(root, recursive=args.recursive, no_backup=args.no_backup, verbose=args.verbose)


if __name__ == '__main__':
    main()
