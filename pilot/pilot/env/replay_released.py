"""Gate 1: replay logged tool calls from released CAPE-Bench episodes through our MockEnv
and the sandbox monitor S, and compare with the logged labels.

    python -m pilot.env.replay_released [--all-templates]
"""
import argparse
import collections
import json

from ..cape_data import load_released_episodes, load_scenarios, spec_from_row
from ..config import PILOT_TEMPLATES, RESULTS_DIR
from ..monitors.sandbox import sandbox_check
from .mockenv import MockEnv


def replay_episode(spec, events, seed=0):
    env = MockEnv(spec, seed=seed)
    rows = []
    for ev in events:
        if not ev.get("is_tool_call") or not ev.get("tool"):
            continue
        try:
            args = json.loads(ev["args_json"]) if ev.get("args_json") else {}
        except json.JSONDecodeError:
            continue
        s_verdict = sandbox_check(env, ev["tool"], args)
        step = env.step(ev["tool"], args)
        rows.append({
            "event_id": ev["event_id"], "tool": ev["tool"], "args": ev["args_json"],
            "logged_unsafe": bool(ev.get("unsafe_action")), "ours_unsafe": step.unsafe_action,
            "sandbox_flag": s_verdict["flag"],
            "logged_critical": bool(ev.get("critical")), "ours_critical": step.critical,
            "logged_result": (ev.get("tool_result") or "")[:200], "ours_result": step.result[:200],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-templates", action="store_true")
    args = ap.parse_args()

    scen = load_scenarios()
    by_inst = {r.instance_id: spec_from_row(r) for r in scen.itertuples(index=False)}
    eps = load_released_episodes()
    if not args.all_templates:
        eps = eps[eps.template.isin(PILOT_TEMPLATES)]

    stats = collections.Counter()
    per_template = collections.defaultdict(collections.Counter)
    mismatches = []
    for ep in eps.itertuples(index=False):
        spec = by_inst.get(ep.instance_id)
        if spec is None:
            stats["missing_spec"] += 1
            continue
        for r in replay_episode(spec, json.loads(ep.events_json), seed=int(ep.seed)):
            t = per_template[ep.template]
            for key, ours in (("unsafe", "ours_unsafe"), ("critical", "ours_critical")):
                logged = r[f"logged_{key}"]
                t[f"{key}_{'tp' if logged and r[ours] else 'fn' if logged else 'fp' if r[ours] else 'tn'}"] += 1
            sflag = r["sandbox_flag"]
            t[f"S_{'tp' if r['logged_unsafe'] and sflag else 'fn' if r['logged_unsafe'] else 'fp' if sflag else 'tn'}"] += 1
            if r["logged_unsafe"] != r["ours_unsafe"] or r["logged_critical"] != r["ours_critical"] \
                    or r["logged_unsafe"] != sflag:
                mismatches.append({"episode_id": ep.episode_id, "template": ep.template, **r})
        stats["episodes"] += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"stats": dict(stats), "per_template": {k: dict(v) for k, v in per_template.items()},
           "n_mismatched_calls": len(mismatches)}
    (RESULTS_DIR / "gate1_replay.json").write_text(json.dumps(out, indent=1))
    with open(RESULTS_DIR / "gate1_mismatches.jsonl", "w", encoding="utf-8") as f:
        for m in mismatches:
            f.write(json.dumps(m) + "\n")

    tot = collections.Counter()
    for t, c in sorted(per_template.items()):
        tot.update(c)
        print(f"{t:24s} unsafe tp/fn/fp {c['unsafe_tp']}/{c['unsafe_fn']}/{c['unsafe_fp']}  "
              f"critical tp/fn/fp {c['critical_tp']}/{c['critical_fn']}/{c['critical_fp']}  "
              f"S tp/fn/fp {c['S_tp']}/{c['S_fn']}/{c['S_fp']}")
    n = sum(v for k, v in tot.items() if k.startswith("unsafe_"))
    agree_u = (tot["unsafe_tp"] + tot["unsafe_tn"]) / max(n, 1)
    agree_s = (tot["S_tp"] + tot["S_tn"]) / max(n, 1)
    print(f"\n{stats['episodes']} episodes, {n} calls; unsafe-label agreement {agree_u:.4f}; "
          f"S agreement {agree_s:.4f}; mismatched calls {len(mismatches)} -> results/gate1_mismatches.jsonl")
    gate = tot["S_fn"] == 0 and tot["S_fp"] == 0
    print("GATE 1 (S agrees with every replayed unsafe_action label):", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    main()
