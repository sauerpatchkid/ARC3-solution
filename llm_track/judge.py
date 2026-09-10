"""judge.py — C2: ask an LLM which of two moves looks more like progress.

Runs in the SEPARATE .venv-llm (vLLM + its own torch), never the baseline venv:

    .venv-llm/bin/python -m llm_track.judge --model Qwen/Qwen3.5-4B
    .venv-llm/bin/python -m llm_track.judge --model Qwen/Qwen3.5-9B

Reads results/llm/probeA/pairs.jsonl (built by probe_pairs.py), asks about every
pair in BOTH orders, and writes labels_<model>.jsonl plus a .meta.json recording
the exact prompt, sampling settings and throughput.

HALLUCINATION-SAFE BY CONSTRUCTION
  Only the letter (A / B / TIE) is consumed. The one-line reason is saved for
  the report and never read by anything downstream, so a made-up reason cannot
  change a label.

POSITION BIAS
  Small models often favour whichever option comes first. Each pair is asked
  twice with the sides swapped; the verdict is a side only if both orders pick
  it, otherwise TIE (design 4.2). An unparseable answer also counts as TIE.

THINKING OFF, TEMPERATURE 0
  Labeling wants speed and determinism. Thinking mode is for the periodic
  heuristic writer (Phase 2), not for thousands of A/B calls.

PROMPT v2: REASON FIRST, LETTER LAST
  v1 asked for the letter first. On hand-made smoke pairs (never the probe
  set) Qwen3.5-4B then answered "A" in BOTH orders — its written reason picked
  the right move, but the letter tracked position, because it committed to a
  letter before writing any reasoning. v2 asks for the reason first. Prompt
  changes are only ever tested on hand-made pairs, never on pairs.jsonl.

The model sees the move texts and nothing else — no game id, no level, no
hint of which side is the anchor.
"""
import argparse
import json
import os
import re
import time

SYSTEM = """You are analysing moves in an abstract grid puzzle game played on a \
64x64 grid of 16 colours. The player does not know the rules or the goal. They \
are experimenting to find out how to complete the level.

Each move is described in a fixed format:
- ACTION1 to ACTION5 are button presses. ACTION6 is a click at a (row, col), \
with the colour of the clicked cell.
- "changed N cells" is how many cells changed colour after the move.
- "ticker region": cells that change on their own, like a timer or move \
counter. Changes there are not caused by what the player chose to do.
- "a shape MOVED by (dy, dx)": a same-shaped group of cells shifted position.
- "compK": a connected group of changed cells, with its size, position, and \
how many cells of each colour it gained (+) or lost (-).
- "counts colour c: a->b": total cells of that colour on the whole grid \
before and after.

You will see two moves from the same game and the same level. Decide which one \
is more likely to be real progress toward completing the level."""

PROMPT_VERSION = 2

# Generation budget. Reason-first replies need room: at 64 tokens, 40 of the
# 4B's 1,588 replies were cut off mid-reasoning before the answer line and
# counted as ties. 192 leaves ample room for "a reason of at most 20 words".
MAX_TOKENS = 192
# Engine sizing for the 9B on a 32 GB card. The 9B's weights take 16.8 GiB, and
# at the defaults (256 sequences, 8192-token batches) activation and CUDA-graph
# memory left no room for cache blocks. Smaller batches free that memory; with
# ~1,600 short prompts throughput is unaffected in practice.
MAX_NUM_SEQS = 64
MAX_BATCHED_TOKENS = 4096

USER = """Move A: {a}
Move B: {b}

Which move is more likely to be progress toward completing the level? Answer \
TIE only if they are equally promising.
Reply with exactly two lines:
line 1: a reason of at most 20 words, comparing the two moves
line 2: your answer, exactly one of: A, B, TIE"""

# The whole last line must be the answer (optionally "line 2:" / "Answer:").
_ANSWER = re.compile(r"^\W*(?:LINE\s*2\s*:)?\W*(?:ANSWER\s*:)?\W*(?:MOVE\s+)?"
                     r"(TIE|A|B)\W*$", re.IGNORECASE)


def _lines(text):
    return [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]


def parse(text):
    """LAST line -> 'A' | 'B' | 'TIE' | 'FAIL'. Only that line is read, and it
    must be nothing but the answer: 'A' is also an English article, so
    searching the reasoning for a letter would misfire."""
    lines = _lines(text)
    m = _ANSWER.match(lines[-1]) if lines else None
    return m.group(1).upper() if m else "FAIL"


def reason(text):
    lines = _lines(text)
    return lines[0] if len(lines) > 1 else ""


def combine(ans_xy, ans_yx):
    """Both orders -> verdict. Order 1 shows x as A; order 2 shows y as A."""
    v1 = {"A": "x", "B": "y"}.get(ans_xy, "tie")
    v2 = {"A": "y", "B": "x"}.get(ans_yx, "tie")
    return v1, v2, (v1 if v1 == v2 else "tie")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id, e.g. Qwen/Qwen3.5-9B")
    ap.add_argument("--pairs", default="results/llm/probeA/pairs.jsonl")
    ap.add_argument("--out", default=None, help="default: next to --pairs")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (smoke test)")
    ap.add_argument("--max-model-len", type=int, default=2048,
                    help="REQUIRED to be small: the models default to 262K context, "
                         "and vLLM would try to reserve KV cache for all of it")
    # 0.90, NOT derived from nvidia-smi: on WSL it reports Windows-side memory
    # CUDA can still use (it showed 6.6 GB "used" while torch measured 30.2 of
    # 31.8 GiB free). vLLM checks against its own measurement and errors
    # clearly if this is too high; lower it if other GPU apps are running.
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    a = ap.parse_args()

    # Greedy decoding never needs a top-k/top-p kernel, but vLLM warms up
    # FlashInfer's sampler anyway, and that JIT-compiles CUDA code -> needs
    # nvcc, which this machine does not have system-wide. Turn it off.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Fallback for any other JIT: torch's wheels ship a pip nvcc (CUDA 13.4,
    # which targets the 5090's sm_120) inside the venv. Point CUDA_HOME at it.
    import sys
    cu = os.path.join(sys.prefix, "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}",
                      "site-packages", "nvidia", "cu13")
    if "CUDA_HOME" not in os.environ and os.path.exists(os.path.join(cu, "bin", "nvcc")):
        os.environ["CUDA_HOME"] = cu
        os.environ["PATH"] = os.path.join(cu, "bin") + os.pathsep + os.environ.get("PATH", "")

    from vllm import LLM, SamplingParams        # only exists in .venv-llm
    import vllm

    with open(a.pairs) as f:
        pairs = [json.loads(ln) for ln in f]
    if a.limit:
        pairs = pairs[:a.limit]
    tag = a.model.split("/")[-1]
    out_dir = a.out or os.path.dirname(a.pairs)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    out_path = os.path.join(out_dir, f"labels_{tag}{suffix}.jsonl")

    # Two conversations per pair: x shown as A, then y shown as A.
    convs = []
    for p in pairs:
        for first, second in ((p["x"], p["y"]), (p["y"], p["x"])):
            convs.append([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER.format(
                              a=first["text"], b=second["text"])}])

    t_load = time.time()
    llm = LLM(model=a.model, dtype="bfloat16", seed=0,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              limit_mm_per_prompt={"image": 0, "video": 0},
              max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_BATCHED_TOKENS)
    load_sec = time.time() - t_load
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    t0 = time.time()
    outs = llm.chat(convs, sp, use_tqdm=True,
                    chat_template_kwargs={"enable_thinking": False})
    gen_sec = time.time() - t0

    n_fail = 0
    with open(out_path, "w") as f:
        for k, p in enumerate(pairs):
            t_xy = outs[2 * k].outputs[0].text
            t_yx = outs[2 * k + 1].outputs[0].text
            a_xy, a_yx = parse(t_xy), parse(t_yx)
            n_fail += (a_xy == "FAIL") + (a_yx == "FAIL")
            v1, v2, verdict = combine(a_xy, a_yx)
            f.write(json.dumps({
                "pair_id": p["pair_id"], "tier": p["tier"], "game": p["game"],
                "event": p["event"], "model": a.model,
                "answers": [a_xy, a_yx], "v_order1": v1, "v_order2": v2,
                "verdict": verdict, "consistent": v1 == v2,
                "reasons": [reason(t_xy), reason(t_yx)],
                "raw": [t_xy, t_yx],
            }) + "\n")

    meta = {
        "model": a.model, "vllm_version": vllm.__version__,
        "n_pairs": len(pairs), "n_calls": len(convs), "parse_failures": n_fail,
        "load_sec": round(load_sec, 1), "gen_sec": round(gen_sec, 1),
        "pairs_per_sec": round(len(pairs) / max(gen_sec, 1e-9), 2),
        "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS},
        "gpu_mem": a.gpu_mem, "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_BATCHED_TOKENS,
        "prompt_version": PROMPT_VERSION,
        "thinking": False, "max_model_len": a.max_model_len,
        "system_prompt": SYSTEM, "user_template": USER,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_path.replace(".jsonl", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[judge] {a.model}: {len(pairs)} pairs x 2 orders in {gen_sec:.0f}s "
          f"({meta['pairs_per_sec']} pairs/s, load {load_sec:.0f}s), "
          f"{n_fail} unparseable of {len(convs)} -> {out_path}")


if __name__ == "__main__":
    main()
