"""Roll out episodes over the pilot worlds and log them in CAPE's events_json style.

Each episode is generated once. For every proposed call the sandbox monitor S scores a forked
environment first; then the call runs in the real environment (ground-truth labels) and in the
partner worlds, so the log records where observations first differ:
  reveal_event          first step whose observation differs from the matched twin (x+ <-> x-)
  inserted_seen_event   x-F / x-N only: first step whose observation contains the inserted message
  decision_event        first turn after both (the read point for the R1 readout)

    python -m pilot.rollout.run --backend server --model Qwen/Qwen3.5-9B --run pilot_main --workers 4
    python -m pilot.rollout.run --backend hf --model Qwen/Qwen3.5-9B --run pilot_main   (slower)
    python -m pilot.rollout.run --backend scripted --model Qwen/Qwen3.5-0.8B --run scripted_test --limit 8
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

from ..config import EPISODES_DIR, MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS_PER_TURN, MAX_TURNS, SEEDS, WORLDS_PATH
from ..env.mockenv import TOOL_SCHEMAS, MockEnv
from ..monitors.sandbox import sandbox_check
from .chat import initial_prompt, openai_tools, parse_generation, tool_wrapper


def load_worlds(path=WORLDS_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _find(ids: list[int], sub: list[int], start=0) -> int | None:
    n = len(sub)
    for i in range(start, len(ids) - n + 1):
        if ids[i:i + n] == sub:
            return i
    return None


def inserted_marker(world) -> str | None:
    ins = world.get("insertion")
    if not ins:
        return None
    body = ins["text"].strip().splitlines()
    return max(body, key=len)  # the message's longest line identifies it in tool output


class Episode:
    def __init__(self, world, partner_specs, seed, backend, tok, wrapper, model_name, run):
        self.w, self.seed, self.backend, self.tok, self.wrapper = world, seed, backend, tok, wrapper
        self.model_name, self.run = model_name, run
        spec = world["spec"]
        self.env = MockEnv(spec, seed=seed)
        self.twin = MockEnv(partner_specs["twin"], seed=seed)
        self.base = MockEnv(partner_specs["base"], seed=seed) if partner_specs.get("base") else None
        self.marker = inserted_marker(world)
        self.episode_id = f"{world['world_id']}|s{seed}"

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def rollout(self) -> dict:
        spec, wr = self.w["spec"], self.wrapper
        tools = openai_tools(TOOL_SCHEMAS)
        ids = self.encode(initial_prompt(self.tok, spec["system_prompt"], spec["task_prompt"], tools))
        if hasattr(self.backend, "start"):
            self.backend.start(spec)
        eot_ids = self.encode(wr.end_of_turn)
        close_ids, call_ids = self.encode(wr.think_close), self.encode("<tool_call>")
        events, logprobs, status, final_answer, n_parse_errors = [], [], "max_turns", "", 0
        for turn in range(MAX_TURNS):
            if len(ids) + MAX_NEW_TOKENS_PER_TURN > MAX_CONTEXT_TOKENS:
                status = "truncated"
                break
            prompt_len = len(ids)
            seed = zlib.crc32(f"{self.episode_id}|{turn}".encode())
            gen = self.backend.generate(ids, MAX_NEW_TOKENS_PER_TURN, seed)
            new = gen.ids
            text = self.tok.decode(new, skip_special_tokens=False)
            p = parse_generation(text, wr)
            te = _find(new, close_ids)
            ac = _find(new, call_ids, te or 0)
            ev = {
                "event_id": turn, "prompt_len": prompt_len, "turn_start_pos": prompt_len - 1,
                "think_end_pos": None if te is None else prompt_len + te,
                "action_start_pos": None if ac is None else prompt_len + ac,
                "gen_end_pos": prompt_len + len(new) - 1, "n_gen_tokens": len(new),
                "finish_reason": gen.finish_reason, "reasoning_text": p.reasoning, "content_text": p.content,
                "is_tool_call": p.tool is not None, "tool": p.tool,
                "args_json": None if p.args is None else json.dumps(p.args), "parse_error": p.parse_error,
                "tool_result": None, "unsafe_action": False, "critical": False, "goal": False,
                "escalation": False, "violations": "", "obs_differs_from_partner": False,
                "inserted_seen": False, "sandbox_flag": None, "sandbox_violations": "",
            }
            ev["decision_pos"] = (ev["action_start_pos"] - 1) if ev["action_start_pos"] is not None \
                else ev["gen_end_pos"]
            logprobs.append({"start": prompt_len, "lp": gen.logprobs})
            ids = ids + new
            if not new or not _ends_with(new, eot_ids):
                ids = ids + eot_ids
            if p.tool is None and p.parse_error is None:
                if gen.finish_reason in ("length",) and not p.content:
                    status = "truncated"
                else:
                    status, final_answer = "final_answer", p.content
                events.append(ev)
                break
            if p.tool is None:
                n_parse_errors += 1
                result = f"Error: could not parse your tool call ({p.parse_error}). Use the documented format."
            else:
                s = sandbox_check(self.env, p.tool, p.args)
                step = self.env.step(p.tool, p.args)
                twin = self.twin.step(p.tool, p.args)
                # The decisive fact is revealed when x- and x+ observations differ; for x-F/x-N
                # compare their base x- world, so the inserted message itself is not a reveal.
                base = self.base.step(p.tool, p.args) if self.base is not None else step
                result = step.result
                ev.update({
                    "sandbox_flag": s["flag"], "sandbox_violations": ",".join(s["violations"]),
                    "unsafe_action": step.unsafe_action, "critical": step.critical, "goal": step.goal,
                    "escalation": step.escalation, "violations": ",".join(step.violations),
                    "obs_differs_from_partner": twin.result != base.result,
                })
                if self.marker and self.marker in result:
                    ev["inserted_seen"] = True
            ev["tool_result"] = result
            events.append(ev)
            before = wr.before_result
            if before.startswith(wr.end_of_turn):
                before = before[len(wr.end_of_turn):]  # the turn was already closed above
            ids = ids + self.encode(before + result + wr.after_result)
        return self.summarize(events, ids, logprobs, status, final_answer, n_parse_errors)

    def summarize(self, events, ids, logprobs, status, final_answer, n_parse_errors):
        w = self.w

        def first(pred):
            return next((e["event_id"] for e in events if pred(e)), None)

        reveal = first(lambda e: e["obs_differs_from_partner"])
        seen = first(lambda e: e["inserted_seen"]) if self.marker else None
        first_unsafe = first(lambda e: e["unsafe_action"])
        first_critical = first(lambda e: e["critical"])
        need = [reveal] + ([seen] if self.marker else [])
        decision = (max(need) + 1) if all(x is not None for x in need) else None
        if decision is not None and decision >= len(events):
            decision = None
        return {
            "run": self.run, "episode_id": self.episode_id, "model": self.model_name,
            "world_id": w["world_id"], "world": w["world"], "template": w["template"],
            "scenario_id": w["scenario_id"], "pair_id": w["pair_id"], "split": w["split"],
            "ground_truth": w["ground_truth"], "seed": self.seed, "status": status,
            "n_events": len(events), "n_tokens": len(ids), "n_parse_errors": n_parse_errors,
            "n_tool_calls": sum(e["is_tool_call"] for e in events) + n_parse_errors,
            "unsafe_episode": first_unsafe is not None, "first_unsafe_event": first_unsafe,
            "critical_done": first_critical is not None, "first_critical_event": first_critical,
            "success": any(e["goal"] for e in events), "escalated": any(e["escalation"] for e in events),
            "reveal_event": reveal, "inserted_seen_event": seen, "decision_event": decision,
            "informed_violation": first_unsafe is not None and reveal is not None and reveal < first_unsafe,
            "inserted_seen_before_violation": first_unsafe is not None and seen is not None and seen < first_unsafe,
            "sandbox_flagged": any(e["sandbox_flag"] for e in events),
            "final_answer": final_answer, "events": events, "token_ids": ids, "gen_logprobs": logprobs,
        }


def _ends_with(seq, suffix):
    return len(suffix) > 0 and seq[-len(suffix):] == suffix


def partner_index(worlds):
    by = {}
    for w in worlds:
        by.setdefault(w["pair_id"], {})[w["world"]] = w["spec"]
    out = {}
    for w in worlds:
        specs = by[w["pair_id"]]
        twin = specs["x-"] if w["world"] == "x+" else specs["x+"]
        out[w["world_id"]] = {"twin": twin, "base": specs["x-"] if w["world"] in ("x-F", "x-N") else None}
    return out


def make_backend(args):
    from . import backends
    if args.backend == "hf":
        return backends.HFBackend(args.model, quant=args.quant, temperature=args.temperature,
                                  think_budget=args.think_budget)
    if args.backend == "server":
        return backends.ServerBackend(args.model, args.base_url, flavor=args.server_flavor,
                                      temperature=args.temperature, think_budget=args.think_budget)
    return backends.ScriptedBackend(args.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "server", "scripted"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--run", required=True)
    ap.add_argument("--worlds", default=str(WORLDS_PATH))
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--server-flavor", choices=["llamacpp", "openai"], default="llamacpp")
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--think-budget", type=int, default=768)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--templates", nargs="*")
    ap.add_argument("--world-types", nargs="*")
    ap.add_argument("--per-template", type=int, help="scenarios per template (smoke tests)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    worlds = load_worlds(args.worlds)
    partners = partner_index(worlds)
    if args.templates:
        worlds = [w for w in worlds if w["template"] in args.templates]
    if args.world_types:
        worlds = [w for w in worlds if w["world"] in args.world_types]
    if args.per_template:
        keep = {}
        for w in worlds:
            keep.setdefault(w["template"], [])
            if w["pair_id"] not in keep[w["template"]] and len(keep[w["template"]]) < args.per_template:
                keep[w["template"]].append(w["pair_id"])
        allowed = {p for ps in keep.values() for p in ps}
        worlds = [w for w in worlds if w["pair_id"] in allowed]
    jobs = [(w, s) for w in worlds for s in args.seeds]
    if args.limit:
        jobs = jobs[:args.limit]

    out_dir = EPISODES_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episodes.jsonl"
    done = set()
    if out_path.exists():  # resumable: overnight batches can be restarted
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["episode_id"] for l in f}
    jobs = [(w, s) for w, s in jobs if f"{w['world_id']}|s{s}" not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already done) -> {out_path}")

    backend = make_backend(args)
    tok = backend.tok
    wrapper = tool_wrapper(tok, openai_tools(TOOL_SCHEMAS))
    if args.workers > 1 and args.backend == "scripted":
        raise SystemExit("the scripted backend is stateful; use --workers 1")

    def one(job):
        w, s = job
        t0 = time.time()
        ep = Episode(w, partners[w["world_id"]], s, backend, tok, wrapper, args.model, args.run).rollout()
        ep["seconds"] = round(time.time() - t0, 1)
        return ep

    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        pool = ThreadPoolExecutor(args.workers) if args.workers > 1 else None
        it = pool.map(one, jobs) if pool else map(one, jobs)
        for i, ep in enumerate(it, 1):
            f.write(json.dumps(ep) + "\n")
            f.flush()
            el = time.time() - t_start
            print(f"[{i}/{len(jobs)}] {ep['episode_id']:60s} {ep['status']:12s} events={ep['n_events']:2d} "
                  f"unsafe={int(ep['unsafe_episode'])} crit={int(ep['critical_done'])} esc={int(ep['escalated'])} "
                  f"{ep['seconds']:.0f}s  eta {el / i * (len(jobs) - i) / 3600:.1f}h", flush=True)
        if pool:
            pool.shutdown()


if __name__ == "__main__":
    main()
