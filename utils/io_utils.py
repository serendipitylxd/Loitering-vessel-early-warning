import os, json
from pathlib import Path

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(str(Path(path).parent))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_files_multi(dir_path, exts):
    files = []
    for e in exts:
        files += sorted([str(p) for p in Path(dir_path).glob(f'*.{e}')])
    return files

