"""advice.py - where a coach's advice comes from, in one shape for every source
(docs/plans/llm-coach.md sections 4.3 and 4.7).

Advice is the JSON the LLM is asked for:
  {"hypothesis": str,
   "click_targets": [{"id": int, "weight": 0-5}, ...],   ids from the summary's table
   "action_weights": {"ACTION1": 0-5, ...},             available buttons only
   "avoid": {"objects": [id, ...], "actions": ["ACTION2", ...]}}

Sources: the LLM (`LLMClient`, OpenAI-compatible HTTP to experiments/coach/serve.sh,
standard library only - the agent never imports vLLM), the hand-written rule
(`heuristic_advice`, arm B2) and random advice (`random_advice`, arm B1, the
placebo). `ranking()` turns any advice into the ordered objects and buttons the
offline probe scores and the bias builder will use.
"""
import hashlib
import json
import os
import random
import time
import urllib.request

PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_v1.txt")
BUTTON_NAMES = [f"ACTION{k}" for k in range(1, 6)]


def system_prompt():
    with open(PROMPT_PATH) as f:
        return f.read().strip()


def prompt_hash():
    return hashlib.sha256(system_prompt().encode()).hexdigest()[:12]


def schema(summary):
    """The JSON schema for one screen: ids limited to the table's objects and
    button names to the buttons available, so a valid answer can't name
    something that isn't there."""
    ids = [o["id"] for o in summary["objects"]] if summary.get("clicks") else []
    names = [BUTTON_NAMES[b["action"]] for b in summary["buttons"]]
    w = {"type": "integer", "minimum": 0, "maximum": 5}
    target = {"type": "object", "properties": {"id": {"type": "integer", "enum": ids or [-1]}, "weight": w},
              "required": ["id", "weight"]}
    return {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string", "maxLength": 300},
            "click_targets": {"type": "array", "maxItems": 5 if ids else 0, "items": target},
            "action_weights": {"type": "object", "properties": {n: w for n in names},
                               "required": names, "additionalProperties": False},
            "avoid": {"type": "object", "properties": {
                "objects": {"type": "array", "maxItems": 10, "items": {"type": "integer", "enum": ids or [-1]}},
                "actions": {"type": "array", "maxItems": 5, "items": {"type": "string", "enum": names or ["none"]}}},
                "required": ["objects", "actions"]},
        },
        "required": ["hypothesis", "click_targets", "action_weights", "avoid"],
    }


def validate(advice, summary):
    """Clean advice against the summary: unknown ids and buttons dropped,
    weights clamped to 0-5. Returns (clean advice, list of problems)."""
    probs = []
    if not isinstance(advice, dict):
        return None, ["not a JSON object"]
    ids = {o["id"] for o in summary["objects"]} if summary.get("clicks") else set()
    names = {BUTTON_NAMES[b["action"]] for b in summary["buttons"]}

    def weight(v):
        try:
            return max(0, min(5, int(v)))
        except (TypeError, ValueError):
            return 0
    targets = []
    for t in advice.get("click_targets") or []:
        if isinstance(t, dict) and t.get("id") in ids:
            targets.append({"id": t["id"], "weight": weight(t.get("weight"))})
        else:
            probs.append(f"unknown click target {t!r}")
    aw = {}
    for k, v in (advice.get("action_weights") or {}).items():
        if k in names:
            aw[k] = weight(v)
        else:
            probs.append(f"unknown button {k!r}")
    av = advice.get("avoid") or {}
    clean = {"hypothesis": str(advice.get("hypothesis", ""))[:300], "click_targets": targets,
             "action_weights": aw,
             "avoid": {"objects": [i for i in av.get("objects") or [] if i in ids],
                       "actions": [a for a in av.get("actions") or [] if a in names]}}
    return clean, probs


def ranking(advice, summary):
    """(objects best first, {button name: weight}) from any advice. Objects:
    click targets by weight (ties keep the answer's order), avoided ones and
    weight-0 ones dropped."""
    avoid = set(advice["avoid"]["objects"])
    ts = [t for t in advice["click_targets"] if t["id"] not in avoid and t["weight"] > 0]
    order = sorted(range(len(ts)), key=lambda i: (-ts[i]["weight"], i))
    seen, objs = set(), []
    for i in order:
        if ts[i]["id"] not in seen:
            seen.add(ts[i]["id"])
            objs.append(ts[i]["id"])
    return objs, dict(advice["action_weights"])


# --------------------------------------------------------------------------------
# The two control arms
# --------------------------------------------------------------------------------
def heuristic_advice(summary):
    """Arm B2, the hand-written rule: untried objects first, in the summary's
    salience order (small, rare colour); then tried objects by their rate of
    reaching new screens. Buttons: weight from their new-screen rate, never-
    pressed buttons get the top weight."""
    objs = summary["objects"] if summary.get("clicks") else []
    untried = [o for o in objs if o["tried"] == 0]
    tried = sorted((o for o in objs if o["tried"] > 0), key=lambda o: (-o["new"] / o["tried"], o["id"]))
    picks = (untried + tried)[:5]
    targets = [{"id": o["id"], "weight": 5 - i} for i, o in enumerate(picks)]
    aw = {}
    for b in summary["buttons"]:
        rate = b["new"] / b["uses"] if b["uses"] else None
        aw[BUTTON_NAMES[b["action"]]] = 5 if rate is None else max(1, min(4, round(1 + 30 * rate)))
    return {"hypothesis": "hand rule: untried, small, rare-coloured objects first",
            "click_targets": targets, "action_weights": aw, "avoid": {"objects": [], "actions": []}}


def random_advice(summary, rng):
    """Arm B1, the placebo: random objects and weights in the same shape."""
    objs = [o["id"] for o in summary["objects"]] if summary.get("clicks") else []
    picks = rng.sample(objs, min(len(objs), rng.randint(1, 5))) if objs else []
    return {"hypothesis": "random",
            "click_targets": [{"id": i, "weight": rng.randint(1, 5)} for i in picks],
            "action_weights": {BUTTON_NAMES[b["action"]]: rng.randint(0, 5) for b in summary["buttons"]},
            "avoid": {"objects": [], "actions": []}}


# --------------------------------------------------------------------------------
# The LLM
# --------------------------------------------------------------------------------
class LLMClient:
    """OpenAI-compatible chat calls to the coach server, with an optional
    response cache (JSON lines) keyed by everything that affects the answer.
    The cache is for offline work and debugging; scored live runs never read
    another run's answers."""

    def __init__(self, url="http://127.0.0.1:8017/v1", model="coach", temperature=0.3, max_tokens=400,
                 thinking=False, cache_path=None, timeout=300):
        self.url, self.model = url.rstrip("/"), model
        self.temperature, self.max_tokens, self.thinking, self.timeout = temperature, max_tokens, thinking, timeout
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    r = json.loads(line)
                    self.cache[r["key"]] = r
        self.server_model = None

    def _served_model(self):
        """The real model behind the alias, so cached answers from a
        different model are never reused."""
        if self.server_model is None:
            with urllib.request.urlopen(self.url + "/models", timeout=30) as r:
                self.server_model = json.load(r)["data"][0].get("root", "?")
        return self.server_model

    def ask(self, summary, seed=0):
        """Advice for one summary. Returns a record: advice (validated, or
        None), raw text, problems, tokens, latency, cached flag."""
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "seed": seed,
                "messages": [{"role": "system", "content": system_prompt()},
                             {"role": "user", "content": summary["text"]}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "advice", "schema": schema(summary)}},
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        key = hashlib.sha256(json.dumps([self._served_model(), body], sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            rec = dict(self.cache[key], cached=True)
        else:
            req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
            u = out.get("usage", {})
            rec = {"key": key, "model": self._served_model(), "raw": out["choices"][0]["message"]["content"] or "",
                   "finish": out["choices"][0].get("finish_reason"), "prompt_tokens": u.get("prompt_tokens"),
                   "completion_tokens": u.get("completion_tokens"),
                   "latency_s": round(time.perf_counter() - t0, 3)}
            self.cache[key] = rec
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            rec = dict(rec, cached=False)
        try:
            parsed = json.loads(rec["raw"])
        except ValueError:
            return dict(rec, advice=None, problems=["invalid JSON"])
        advice, probs = validate(parsed, summary)
        return dict(rec, advice=advice, problems=probs)
