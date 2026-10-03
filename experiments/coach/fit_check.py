#!/usr/bin/env python3
"""fit_check.py - Coach step 2: does the in-loop LLM fit beside one Goose run on
this GPU, and how fast does it answer a Coach-sized request?

Needs the server from experiments/coach/serve.sh already running. Standard
library only (the coach will talk to the server the same way: HTTP, no vLLM).

  uv run python experiments/coach/fit_check.py              # server from serve.sh on :8017
  uv run python experiments/coach/fit_check.py --label qwen38_27b

Three phases, GPU memory sampled throughout (nvidia-smi, whole card):
  1. Goose beside an IDLE server: one mb_gated_att run (tu93, 10k actions, the
     same game and length as the confirm doc's single-run speed check).
  2. Requests with Goose PAUSED (SIGSTOP; its memory stays allocated). This is
     how the coach runs: calls are synchronous and Goose waits.
  3. Requests with Goose TRAINING. Only matters if runs are ever overlapped.

Each request is Coach-shaped: a ~1.5k-token object table, JSON output forced by
the schema in docs/plans/llm-coach.md section 4.3, max 400 tokens, thinking off.
Writes results/coach/fit_<label>_<stamp>.json.
"""
import argparse
import json
import os
import random
import signal
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOOSE_ENV = {"EVAL_LABEL": "novel", "EVAL_RETURN_MAP": "1",
             "EVAL_UPGRADES": "bars,map_gated,attempt"}

SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "maxLength": 300},
        "click_targets": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "properties": {"id": {"type": "integer"},
                           "weight": {"type": "integer", "minimum": 0, "maximum": 5}},
            "required": ["id", "weight"]}},
        "action_weights": {"type": "object", "properties": {
            f"ACTION{k}": {"type": "integer", "minimum": 0, "maximum": 5} for k in range(1, 6)}},
        "avoid": {"type": "object", "properties": {
            "objects": {"type": "array", "items": {"type": "integer"}},
            "actions": {"type": "array", "items": {"type": "string"}}}},
    },
    "required": ["hypothesis", "click_targets", "action_weights", "avoid"],
}

SYSTEM = ("You are helping an exploration agent that is stuck in an unknown 64x64 grid game. "
          "You see a summary of the objects on screen and what the agent has already tried. "
          "Your job is to pick the most promising things to try next. Prefer untried objects and "
          "actions; prefer small, distinctive, button-like objects; prefer actions whose effects are "
          "unknown over ones proven useless; use the previous level's win, if given, to guess what "
          "progress looks like. Answer with JSON only.")


def fake_summary(rng):
    """A synthetic summary the size of a real one (~1.5k tokens). Step 3 builds
    the real serializer; this only has to be the right length and shape."""
    effects = ["no-op", "changed 2 cells", "changed 36 cells; colours 9->12", "changed 4 cells; colours 3->8"]
    rows = ["id | colour | bbox (x0,y0,x1,y1) | size | same-shape | clicks tried | changed | new screen | last effect"]
    for i in range(30):
        x, y = rng.randrange(60), rng.randrange(60)
        t = rng.randrange(0, 12)
        ch = rng.randrange(0, t + 1)
        rows.append(f"{i} | {rng.randrange(16)} | ({x},{y},{x + 3},{y + 3}) | {rng.randrange(1, 40)} | "
                    f"{rng.randrange(0, 4)} | {t} | {ch} | {rng.randrange(0, ch + 1)} | {rng.choice(effects)}")
    buttons = ["button | uses (last 200) | changed% | new% | dominant effect"]
    for k in range(1, 6):
        buttons.append(f"ACTION{k} | {rng.randrange(200)} | {rng.randrange(100)} | {rng.randrange(30)} | "
                       f"colour {rng.randrange(16)} moved by ({rng.randrange(-5, 6)}, {rng.randrange(-5, 6)})")
    ctx = (f"Level 2. Moves in this level: 14,210. Screens seen: 311. Moves since the last new screen: 1,532. "
           f"Available actions: ACTION1-ACTION4, ACTION6 (click).\n"
           f"Previous level's win: clicking object 'colour 8, 3x3' after the four corner tiles matched; "
           f"colours whose counts changed at the win: 9 (-12), 12 (+12).")
    return "OBJECTS ON SCREEN\n" + "\n".join(rows) + "\n\nBUTTONS\n" + "\n".join(buttons) + "\n\nCONTEXT\n" + ctx


def ask(url, summary):
    body = {"model": "coach", "temperature": 0.3, "max_tokens": 400, "seed": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": summary}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "advice", "schema": SCHEMA}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    dt = time.perf_counter() - t0
    text = out["choices"][0]["message"]["content"] or ""
    try:
        json.loads(text)
        valid = True
    except ValueError:
        valid = False
    u = out.get("usage", {})
    return {"latency_s": round(dt, 3), "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "json_valid": valid}


class MemSampler(threading.Thread):
    """Peak whole-card memory (MiB) per phase, sampled every 0.25 s."""
    def __init__(self):
        super().__init__(daemon=True)
        self.phase, self.peak, self.stop = "start", {}, False

    def run(self):
        while not self.stop:
            try:
                used = int(subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
                self.peak[self.phase] = max(self.peak.get(self.phase, 0), used)
            except Exception:
                pass
            time.sleep(0.25)


class Goose:
    """One mb_gated_att run as a subprocess, with its progress lines parsed."""
    def __init__(self, game, cap, out_dir):
        env = dict(os.environ, EVAL_SEED="0", PYTHONHASHSEED="0", EVAL_MAX_ACTIONS=str(cap),
                   EVAL_RESULTS_DIR=out_dir, **GOOSE_ENV)
        self.p = subprocess.Popen([sys.executable, "-u", "run_local.py", "--game", game], cwd=ROOT,
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.actions, self.aps, self.lines = 0, None, []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.lines.append(line.rstrip())
            s = line.split()
            if len(s) >= 2 and s[1] == "actions" and s[0].isdigit():
                self.actions = int(s[0])
            if "[run_local] done:" in line:
                self.aps = float(line.split("(")[1].split()[0])

    def wait_actions(self, n, timeout=600):
        t0 = time.time()
        while self.actions < n and self.p.poll() is None and time.time() - t0 < timeout:
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8017/v1")
    ap.add_argument("--label", default="qwen38_27b")
    ap.add_argument("--game", default="tu93")
    ap.add_argument("--requests", type=int, default=20)
    a = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "results", "coach", f"fit_runs_{a.label}_{stamp}")
    rng = random.Random(0)
    summaries = [fake_summary(rng) for _ in range(a.requests)]

    mem = MemSampler()
    mem.start()
    time.sleep(1.0)
    mem.phase = "server_idle"
    warm = ask(a.url, summaries[0])                       # first call compiles; not timed
    print(f"warm-up call: {warm}")
    time.sleep(1.0)

    print("phase 1: Goose beside the idle server (tu93, 10k actions)")
    mem.phase = "goose_beside_idle_server"
    g = Goose(a.game, 10_000, out_dir)
    g.p.wait()
    time.sleep(1.0)
    if g.aps is None:
        print("\n".join(g.lines[-30:]))
        sys.exit("Goose run failed beside the server (see its output above)")
    print(f"  Goose: {g.aps} act/s")

    print("phase 2: requests with Goose paused")
    mem.phase = "goose_paused_plus_requests"
    g2 = Goose(a.game, 1_000_000, out_dir + "_2")
    g2.wait_actions(3000)
    os.kill(g2.p.pid, signal.SIGSTOP)
    paused = [ask(a.url, s) for s in summaries]
    os.kill(g2.p.pid, signal.SIGCONT)

    print("phase 3: requests with Goose training")
    mem.phase = "goose_training_plus_requests"
    a0 = g2.actions
    t0 = time.time()
    running = [ask(a.url, s) for s in summaries]
    goose_aps_during = (g2.actions - a0) / max(time.time() - t0, 1e-9)
    g2.p.terminate()
    g2.p.wait()
    mem.stop = True

    def stats(rs):
        lat = sorted(r["latency_s"] for r in rs)
        return {"n": len(rs), "latency_median_s": round(st.median(lat), 3),
                "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))], 3),
                "prompt_tokens_median": st.median(r["prompt_tokens"] for r in rs),
                "completion_tokens_median": st.median(r["completion_tokens"] for r in rs),
                "json_valid": sum(r["json_valid"] for r in rs)}

    rep = {"label": a.label, "url": a.url, "stamp": stamp,
           "card_total_mib": int(subprocess.check_output(
               ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).split()[0]),
           "peak_mib_by_phase": mem.peak,
           "goose_aps_beside_idle_server": g.aps,
           "goose_aps_while_requests_run": round(goose_aps_during, 1),
           "requests_goose_paused": stats(paused), "requests_goose_training": stats(running),
           "raw": {"paused": paused, "training": running}}
    path = os.path.join(ROOT, "results", "coach", f"fit_{a.label}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "raw"}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
