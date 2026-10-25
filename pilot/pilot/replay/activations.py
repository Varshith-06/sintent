"""Teacher-forced replay: one forward pass per episode over its logged token ids, with hooks on
the chosen decoder layers that keep only the positions the readouts need.

Per episode we save, for each of five depths, the residual stream (decoder-layer output)
mean-pooled over the 8 tokens ending at each read position:
  R1  decision state: the last prompt token of the decision turn (turn_start_pos)
  R2  pre-action state of every tool-call turn: the token before the tool-call markup (decision_pos)

    python -m pilot.replay.activations --run pilot_main --model Qwen/Qwen3.5-9B
    python -m pilot.replay.activations --run pilot_main --model Qwen/Qwen3.5-9B --equivalence 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..config import ACTS_DIR, EPISODES_DIR

POOL = 8
DEPTH_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7)


def load_episodes(run):
    with open(EPISODES_DIR / run / "episodes.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def decoder_layers(model):
    """The ModuleList of decoder blocks, wherever the architecture keeps it."""
    import torch.nn as nn
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers") and "vision" not in name and "visual" not in name:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not find decoder layers")
    return best[1]


def read_positions(ep) -> dict[str, int]:
    pos = {}
    ev = {e["event_id"]: e for e in ep["events"]}
    d = ep.get("decision_event")
    if d is not None and d in ev:
        pos["R1"] = ev[d]["turn_start_pos"]
    for e in ep["events"]:
        if e["is_tool_call"] and e.get("action_start_pos") is not None:
            pos[f"R2_{e['event_id']}"] = e["decision_pos"]
    return pos


class Replayer:
    def __init__(self, model_id, quant="4bit"):
        from ..rollout.backends import HFBackend
        b = HFBackend(model_id, quant=quant)
        self.model, self.tok, self.torch = b.model, b.tok, b.torch
        layers = decoder_layers(self.model)
        n = len(layers)
        self.layer_ids = sorted({min(n - 1, round(f * n)) for f in DEPTH_FRACTIONS})
        self.captured = {}
        for li in self.layer_ids:
            layers[li].register_forward_hook(self._hook(li))
        self.want = []

    def _hook(self, li):
        def fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            rows = [h[0, max(0, p - POOL + 1):p + 1].float().mean(0) for p in self.want]
            self.captured[li] = self.torch.stack(rows).cpu().numpy() if rows else None
        return fn

    def forward(self, ids, positions: list[int], want_logprobs=False):
        torch = self.torch
        self.want, self.captured = positions, {}
        end = (max(positions) + 1) if (positions and not want_logprobs) else len(ids)
        inp = torch.tensor([ids[:end]], device=self.model.device)
        final = {}
        handle = None
        if want_logprobs:  # full-vocab logits for 8k tokens do not fit in 8 GB: keep the final hidden states
            norm = self.model.get_decoder().norm
            handle = norm.register_forward_hook(lambda m, i, o: final.__setitem__("h", o))
        with torch.no_grad():
            self.model(input_ids=inp, use_cache=False, logits_to_keep=1)
            lps = None
            if want_logprobs:
                handle.remove()
                head, h = self.model.get_output_embeddings(), final["h"][0]
                chunks = []
                for s in range(0, h.shape[0] - 1, 512):
                    logits = head(h[s:min(s + 512, h.shape[0] - 1)]).float()
                    nxt = inp[0, s + 1:s + 1 + logits.shape[0], None]
                    chunks.append(torch.log_softmax(logits, -1).gather(1, nxt)[:, 0])
                lps = torch.cat(chunks).cpu().numpy()  # lps[i] = log p(token i+1 | tokens <= i)
        feats = np.stack([self.captured[li] for li in self.layer_ids], 1) if positions else None  # [P, L, H]
        return feats, lps


def equivalence(rep, eps, k):
    """Mean |logprob(rollout) - logprob(teacher-forced)| over generated tokens of k episodes."""
    diffs = []
    for ep in eps[:k]:
        _, lps = rep.forward(ep["token_ids"], [], want_logprobs=True)
        for seg in ep["gen_logprobs"]:
            for j, lp in enumerate(seg["lp"]):
                pos = seg["start"] + j          # token index; its logprob is predicted at pos-1
                if lp is not None and 0 < pos <= len(lps):
                    diffs.append(abs(lp - float(lps[pos - 1])))
    return float(np.mean(diffs)) if diffs else float("nan"), len(diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    ap.add_argument("--equivalence", type=int, default=0, help="only run the replay-equivalence check on N episodes")
    args = ap.parse_args()

    eps = load_episodes(args.run)
    rep = Replayer(args.model, args.quant)
    if args.equivalence:
        mad, n = equivalence(rep, [e for e in eps if any(s["lp"] for s in e["gen_logprobs"])], args.equivalence)
        print(f"replay equivalence: mean |dlogprob| = {mad:.4f} over {n} generated tokens "
              f"(the CAPE authors found ~0.01 between vLLM and HF)")
        out = EPISODES_DIR / args.run / "replay_equivalence.json"
        out.write_text(json.dumps({"mean_abs_logprob_diff": mad, "n_tokens": n, "n_episodes": args.equivalence}))
        return

    out_dir = ACTS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"layers": rep.layer_ids, "pool": POOL, "model": args.model, "episodes": {}}
    for i, ep in enumerate(eps, 1):
        pos = read_positions(ep)
        if not pos:
            continue
        keys = list(pos)
        feats, _ = rep.forward(ep["token_ids"], [pos[k] for k in keys])
        fname = f"{i:05d}.npz"
        np.savez_compressed(out_dir / fname, feats=feats.astype(np.float16), keys=np.array(keys))
        meta["episodes"][ep["episode_id"]] = fname
        if i % 20 == 0:
            print(f"[{i}/{len(eps)}] replayed", flush=True)
    (out_dir / "index.json").write_text(json.dumps(meta))
    print(f"saved activations for {len(meta['episodes'])} episodes, layers {rep.layer_ids} -> {out_dir}")


if __name__ == "__main__":
    main()
