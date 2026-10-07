"""manifest.py — the one reader for sweep manifests.

A manifest is what every sweep writes as it goes, one line per finished run:

    run_dir <TAB> game <TAB> seed <TAB> arm

(sweep.sh, experiments/upgrade_screen/screen.py and the scripts built on it).
Seven scripts used to parse this themselves, each a little differently: some
skipped a short line, some crashed on it. They all call read_manifest() now.

    rows = read_manifest(path)                  # [{run_dir, game, seed, arm}, ...]
    rows = read_manifest(path, strict=True)     # a malformed line is an error

`seed` is returned as the text in the file ("0"); callers that sort or print
it as a number convert it themselves. Tolerant mode (the default) skips blank
and short lines, which is what resuming a sweep after a crash needs: a line cut
off mid-write counts as "not done". Strict mode is for tools that give a
verdict, where a line that cannot be read must not be quietly left out.
"""


def read_manifest(path, strict=False):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4 or (strict and len(parts) != 4):
                if strict:
                    raise ValueError(f"{path}:{n}: expected 4 tab-separated fields "
                                     f"(run_dir, game, seed, arm), got {len(parts)}")
                continue
            rows.append({"run_dir": parts[0], "game": parts[1],
                         "seed": parts[2], "arm": parts[3]})
    return rows


def arm_name(arm):
    """How an arm is shown in reports: 'on'/'off' are the reset arms; anything
    else (Plan B's A0, A1, an upgrade name) is already a name."""
    return f"reset_{arm}" if arm in ("on", "off") else str(arm)
