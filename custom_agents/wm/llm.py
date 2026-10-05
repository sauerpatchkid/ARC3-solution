"""llm.py - the Rulebook's LLM calls: OpenAI-compatible chat over HTTP to the
server in experiments/rulebook/serve.sh. Standard library only; the agent's
environment never imports vLLM.

A request asks for n candidates at once. Answers are cached on disk (JSON lines)
keyed by the served model and the whole request, so an offline run can be
resumed or re-scored without the GPU. The cache is for offline work and
debugging; scored live runs never read another run's answers.

`mock=fn` answers from fn(messages, n) instead (tests): a list of texts, or of
{"text", "finish", "tokens"} dicts to imitate a truncated answer.
"""
import hashlib
import json
import os
import threading
import time
import urllib.request


class ChatClient:
    def __init__(self, url="http://127.0.0.1:8018/v1", model="rulebook", cache_path=None,
                 timeout=7200, mock=None):
        self.url, self.model, self.timeout, self.mock = url.rstrip("/"), model, timeout, mock
        self.cache_path, self.cache, self.lock = cache_path, {}, threading.Lock()
        self.meter = {"requests": 0, "cached": 0, "completions": 0, "tokens": 0, "seconds": 0.0}
        self._served = "mock" if mock else None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    r = json.loads(line)
                    self.cache[r["key"]] = r

    def served_model(self):
        """The real model behind the alias: a cached answer from another model is never reused."""
        if self._served is None:
            with urllib.request.urlopen(self.url + "/models", timeout=30) as r:
                self._served = json.load(r)["data"][0].get("root", "?")
        return self._served

    def complete(self, messages, n=1, max_tokens=4096, thinking=True, seed=0, **sampling):
        """n answers to one conversation: [{"text", "finish", "tokens"}]. `text`
        is the reasoning (if any) + "</think>" + the answer, so callers can split
        on </think> whichever way the server returns thinking."""
        body = {"model": self.model, "messages": messages, "n": n, "max_tokens": max_tokens, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)}, **sampling}
        key = hashlib.sha256(json.dumps([self.served_model(), body], sort_keys=True).encode()).hexdigest()
        with self.lock:
            hit = self.cache.get(key)
            self.meter["requests"] += 1
        if hit is not None:
            with self.lock:
                self.meter["cached"] += 1
            return hit["outputs"]
        t0 = time.perf_counter()
        if self.mock:
            outs = [t if isinstance(t, dict) else {"text": t, "finish": "stop", "tokens": len(t) // 4}
                    for t in self.mock(messages, n)]
        else:
            req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
            per = (out.get("usage", {}).get("completion_tokens") or 0) // max(len(out["choices"]), 1)
            outs = []
            for ch in sorted(out["choices"], key=lambda c: c.get("index", 0)):
                m = ch["message"]
                think = m.get("reasoning_content") or m.get("reasoning") or ""
                text = (think + "</think>" if think else "") + (m.get("content") or "")
                outs.append({"text": text, "finish": ch.get("finish_reason"), "tokens": per})
        rec = {"key": key, "model": self.served_model(), "outputs": outs,
               "seconds": round(time.perf_counter() - t0, 2)}
        with self.lock:
            self.cache[key] = rec
            self.meter["completions"] += len(outs)
            self.meter["tokens"] += sum(o["tokens"] for o in outs)
            self.meter["seconds"] += rec["seconds"]
            if self.cache_path:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return outs
