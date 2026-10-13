"""Small, deterministic experiment artifacts; no implicit overwrite of results."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def assert_frozen(expected):
    from .train import source_hashes
    if expected != source_hashes():
        raise ValueError("Package source differs from the frozen experiment. Use its saved "
                         "source snapshot or prepare a NEW study; do not mix versions.")
