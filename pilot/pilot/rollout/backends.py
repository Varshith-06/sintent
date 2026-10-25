"""Generation backends. Each takes the transcript as token ids and returns new token ids.

- HFBackend: transformers + bitsandbytes 4-bit on the local GPU (works natively on Windows).
  Records the raw-logit log-probability of every sampled token for the replay-equivalence check.
- ServerBackend: any OpenAI-compatible /v1/completions server that accepts token-id prompts
  (vLLM under WSL2/Linux, llama.cpp's server). Faster with --workers > 1.
- ScriptedBackend: a deterministic agent for CPU tests; no model needed.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Generation:
    ids: list[int]
    logprobs: list[float | None] = field(default_factory=list)
    finish_reason: str = "stop"


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


class HFBackend:
    name = "hf"

    def __init__(self, model_id: str, quant: str = "4bit", temperature=0.6, top_p=0.95, top_k=20,
                 think_budget: int = 768):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        self.torch = torch
        self.tok = load_tokenizer(model_id)
        kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
        if quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        self.model.eval()
        self.sampling = dict(do_sample=temperature > 0, temperature=temperature or None, top_p=top_p, top_k=top_k)
        self.think_budget = think_budget
        self.stop_ids = sorted({i for i in [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")]
                                if isinstance(i, int) and i >= 0})
        self.think_close_ids = self.tok.encode("</think>", add_special_tokens=False)

    def _gen(self, ids, max_new, stopping=None):
        torch = self.torch
        inp = torch.tensor([ids], device=self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inp, attention_mask=torch.ones_like(inp), max_new_tokens=max_new,
                eos_token_id=self.stop_ids, pad_token_id=self.stop_ids[0], return_dict_in_generate=True,
                output_logits=True, stopping_criteria=stopping, **self.sampling)
        new = out.sequences[0, len(ids):].tolist()
        lps = []
        for t, logits in zip(new, out.logits):
            lps.append(float(torch.log_softmax(logits[0].float(), -1)[t]))
        return new, lps

    def generate(self, ids: list[int], max_new_tokens: int, seed: int) -> Generation:
        from transformers import StoppingCriteria, StoppingCriteriaList
        self.torch.manual_seed(seed)
        n0, close = len(ids), self.think_close_ids
        budget = self.think_budget

        class ThinkBudget(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                gen = input_ids[0, n0:].tolist()
                return len(gen) >= budget and not _contains(gen, close)

        new, lps = self._gen(ids, max_new_tokens, StoppingCriteriaList([ThinkBudget()]))
        if len(new) >= budget and not _contains(new, close) and (not new or new[-1] not in self.stop_ids):
            # Reasoning ran past its budget: close it and let the model act.
            forced = self.tok.encode("\n</think>\n\n", add_special_tokens=False)
            more, more_lps = self._gen(ids + new + forced, max(max_new_tokens - len(new), 64))
            return Generation(new + forced + more, lps + [None] * len(forced) + more_lps, "think_budget")
        reason = "stop" if new and new[-1] in self.stop_ids else "length"
        return Generation(new, lps, reason)


def _contains(seq, sub):
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))


class ServerBackend:
    """flavor='llamacpp': llama.cpp's native /completion, which returns the sampled token ids and
    reuses the cached prompt prefix across turns. flavor='openai': /v1/completions (vLLM); the
    returned text is re-tokenized, which can shift token boundaries (checked by replay equivalence)."""
    name = "server"

    def __init__(self, model_id: str, base_url: str, flavor="llamacpp", temperature=0.6, top_p=0.95, top_k=20,
                 think_budget: int = 768):
        self.tok = load_tokenizer(model_id)
        self.model_id, self.base_url, self.flavor = model_id, base_url.rstrip("/"), flavor
        self.params = dict(temperature=temperature, top_p=top_p, top_k=top_k)
        self.think_budget = think_budget
        self.close_ids = self.tok.encode("</think>", add_special_tokens=False)
        self.eot_ids = self.tok.encode("<|im_end|>", add_special_tokens=False)

    def _post(self, path, body):
        req = urllib.request.Request(f"{self.base_url}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)

    def _gen(self, ids, n, seed):
        if self.flavor == "llamacpp":
            out = self._post("/completion", {"prompt": ids, "n_predict": n, "seed": seed, "cache_prompt": True,
                                             "return_tokens": True, "n_probs": 1, "special": True, **self.params})
            new = list(out.get("tokens") or self.tok.encode(out["content"], add_special_tokens=False))
            probs = out.get("completion_probabilities") or []
            lps = [p.get("logprob") for p in probs] if len(probs) == len(new) else [None] * len(new)
            stopped = out.get("stop_type") in ("eos", "word") or bool(out.get("stopped_eos"))
            if stopped and (not new or new[-len(self.eot_ids):] != self.eot_ids):
                new = new + self.eot_ids  # llama.cpp drops the end-of-turn token it stopped on
                lps = lps + [None] * len(self.eot_ids)
            return new, lps, "stop" if stopped else "length"
        out = self._post("/v1/completions", {"model": self.model_id, "prompt": ids, "max_tokens": n, "seed": seed,
                                             "stop": ["<|im_end|>"], "skip_special_tokens": False, "logprobs": 1,
                                             **self.params})
        ch = out["choices"][0]
        new = self.tok.encode(ch["text"], add_special_tokens=False)
        lp = (ch.get("logprobs") or {}).get("token_logprobs") or []
        lps = lp if len(lp) == len(new) else [None] * len(new)  # re-tokenization changed the count
        fin = ch.get("finish_reason") or "stop"
        if fin == "stop":
            new, lps = new + self.eot_ids, lps + [None] * len(self.eot_ids)
        return new, lps, fin

    def generate(self, ids: list[int], max_new_tokens: int, seed: int) -> Generation:
        budget = min(self.think_budget, max_new_tokens)
        new, lps, fin = self._gen(ids, budget, seed)
        if fin == "length" and not _contains(new, self.close_ids):
            # Reasoning ran past its budget: close it and let the model act.
            forced = self.tok.encode("\n</think>\n\n", add_special_tokens=False)
            more, more_lps, fin = self._gen(ids + new + forced, max(max_new_tokens - len(new), 64), seed + 1)
            return Generation(new + forced + more, lps + [None] * len(forced) + more_lps,
                              "think_budget" if fin == "stop" else fin)
        if fin == "length" and len(new) < max_new_tokens:
            more, more_lps, fin = self._gen(ids + new, max_new_tokens - len(new), seed + 1)
            new, lps = new + more, lps + more_lps
        return Generation(new, lps, fin)


class ScriptedBackend:
    """Lists and reads every file and table, then takes the scenario's example critical action and stops.
    Emits Qwen-style XML tool calls. Used to test the pipeline end to end on CPU."""
    name = "scripted"

    def __init__(self, model_id: str):
        self.tok = load_tokenizer(model_id)
        self.plan = None

    def start(self, spec):
        files = sorted((spec.get("files") or {}).keys())
        tables = sorted((spec.get("db") or {}).keys())
        tool, args = spec["critical_example"]
        self.plan = [("list_files", {"directory": "/"})] + [("read_file", {"path": p}) for p in files] + \
                    [("query_database", {"query": f"SELECT * FROM {t}"}) for t in tables] + [(tool, args), None]

    def generate(self, ids, max_new_tokens, seed) -> Generation:
        step = self.plan.pop(0)
        if step is None:
            text = "I checked the records.\n</think>\n\nDone. I completed the requested change.<|im_end|>"
        else:
            tool, args = step
            params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                             for k, v in args.items())
            text = (f"Next I will call {tool}.\n</think>\n\n<tool_call>\n<function={tool}>\n{params}"
                    f"</function>\n</tool_call><|im_end|>")
        return Generation(self.tok.encode(text, add_special_tokens=False), [], "stop")
