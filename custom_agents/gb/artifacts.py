"""artifacts.py - the artifact contract: how Glassbox's stages stay separate.

Every stage reads files and writes files (plan section 4). One run is one folder:

    results/glassbox/<run>/
      manifest.json            when it was made, and every session that wrote to it
                               (time, git commit, config hash)
      config.toml              the settings the latest session used (a copy)
      s1_see/ ... s6_show/     one folder per stage
        _headers/<game>.json   what produced that stage's files for one game

A header records the stage, its VERSION, the settings it read, and the hash of
every input file. Those together make the stage's KEY. If the header on disk
has the same key, that stage's output for that game is current and the stage is
skipped. So changing a guard-fitter knob (S3's settings) re-runs S3 and the
stages after it, and S1's and S2's files are reused from disk.

Bump a stage's VERSION by hand when what its output MEANS changes (not for a
refactor). That invalidates the stage and everything after it.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repo
DEFAULT_CONFIG = os.path.join(ROOT, "experiments", "glassbox", "glassbox.toml")
STAGES = ("s1_see", "s2_write", "s3_check", "s4_carry", "s5_act", "s6_show")


def results_dir():
    """results/glassbox, under EVAL_RESULTS_DIR like every other run output."""
    base = os.getenv("EVAL_RESULTS_DIR", "results")
    return os.path.join(base if os.path.isabs(base) else os.path.join(ROOT, base), "glassbox")


# ---- hashes ---------------------------------------------------------------------------
def digest(obj):
    """A short, stable hash of anything JSON can hold (settings, input lists)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---- settings ---------------------------------------------------------------------------
def load_config(path=DEFAULT_CONFIG):
    """experiments/glassbox/glassbox.toml as a dict, one section per stage."""
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---- a run -------------------------------------------------------------------------------
class Run:
    """One run folder. Stages ask it whether their output is current and record
    a header when they finish one game."""

    def __init__(self, name, config_path=DEFAULT_CONFIG, root=None):
        self.name = name
        self.dir = os.path.join(root or results_dir(), name)
        self.config_path = config_path
        self.config = load_config(config_path)
        os.makedirs(self.dir, exist_ok=True)
        self._record_session()

    def _record_session(self):
        shutil.copyfile(self.config_path, os.path.join(self.dir, "config.toml"))
        path = os.path.join(self.dir, "manifest.json")
        manifest = json.load(open(path)) if os.path.exists(path) else {
            "run": self.name, "created": _now(), "sessions": []}
        manifest["sessions"].append({"time": _now(), "commit": git_commit(),
                                     "config_hash": digest(self.config)})
        _write_json(path, manifest)

    def stage_dir(self, stage):
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; stages are {', '.join(STAGES)}")
        d = os.path.join(self.dir, stage)
        os.makedirs(d, exist_ok=True)
        return d

    def _header_path(self, stage, game):
        return os.path.join(self.stage_dir(stage), "_headers", f"{game}.json")

    def key(self, stage, version, settings, inputs):
        """The stage's key for one game: its version, the settings it read, and
        the hashes of its input files ({label: path})."""
        return digest({"stage": stage, "version": version, "settings": settings,
                       "inputs": {label: file_sha256(p) for label, p in sorted(inputs.items())}})

    def is_current(self, stage, game, version, settings, inputs):
        """True if this stage already produced its files for this game from
        exactly these inputs, settings and version."""
        h = self.read_header(stage, game)
        return h is not None and h["key"] == self.key(stage, version, settings, inputs)

    def write_header(self, stage, game, version, settings, inputs, outputs):
        """Record what produced this stage's files for one game. Call it after
        the outputs ({label: path}) are written."""
        header = {"stage": stage, "game": game, "version": version, "settings": settings,
                  "inputs": {label: {"path": _rel(p), "sha256": file_sha256(p)}
                             for label, p in sorted(inputs.items())},
                  "outputs": {label: {"path": _rel(p), "sha256": file_sha256(p)}
                              for label, p in sorted(outputs.items())},
                  "key": self.key(stage, version, settings, inputs),
                  "commit": git_commit(), "created": _now()}
        path = self._header_path(stage, game)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_json(path, header)
        return header

    def read_header(self, stage, game):
        path = self._header_path(stage, game)
        return json.load(open(path)) if os.path.exists(path) else None

    def require(self, stage, game, version):
        """For a consumer (a later stage, or S6): the header of a finished stage,
        refusing one written by a different version of that stage."""
        h = self.read_header(stage, game)
        if h is None:
            raise FileNotFoundError(f"{self.name}: {stage} has not run for {game}")
        if h["version"] != version:
            raise ValueError(f"{self.name}: {stage} for {game} is version {h['version']}, "
                             f"this code reads version {version}; re-run {stage}")
        return h


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _rel(path):
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def main():
    """Print the settings file and its hashes: which settings would a run use?"""
    import argparse
    ap = argparse.ArgumentParser(description="Glassbox settings and their hashes")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    a = ap.parse_args()
    config = load_config(a.config)
    print(f"{_rel(a.config)}  config hash {digest(config)}")
    for section, values in config.items():
        print(f"  [{section}]  {digest(values)}  " + ", ".join(f"{k}={v}" for k, v in values.items()))


if __name__ == "__main__":
    main()
