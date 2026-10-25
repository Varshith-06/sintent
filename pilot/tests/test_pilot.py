"""Pilot tests: run with `python -m pytest tests -q` from pilot/."""
import copy
import json

import pytest

from pilot.cape_data import load_released_episodes, load_scenarios, spec_from_row
from pilot.config import PILOT_TEMPLATES
from pilot.env.mockenv import MockEnv
from pilot.env.replay_released import replay_episode
from pilot.monitors.sandbox import sandbox_check
from pilot.worlds.build_worlds import APPROVAL_RULE, build


@pytest.fixture(scope="module")
def worlds():
    return build(n_scenarios=4, strength="plain")


@pytest.fixture(scope="module")
def scenarios():
    return {r.instance_id: spec_from_row(r) for r in load_scenarios().itertuples(index=False)}


def test_sandbox_matches_released_labels(scenarios):
    """Gate 1: S and the env labeler agree with every logged unsafe_action in the pilot templates."""
    eps = load_released_episodes()
    eps = eps[eps.template.isin(PILOT_TEMPLATES)]
    n = 0
    for ep in eps.itertuples(index=False):
        for r in replay_episode(scenarios[ep.instance_id], json.loads(ep.events_json), seed=int(ep.seed)):
            assert r["sandbox_flag"] == r["logged_unsafe"], (ep.episode_id, r)
            assert r["ours_unsafe"] == r["logged_unsafe"], (ep.episode_id, r)
            assert r["ours_critical"] == r["logged_critical"], (ep.episode_id, r)
            n += 1
    assert n > 1000


def _diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in set(a) | set(b):
            out += _diff(a.get(k), b.get(k), f"{path}/{k}")
        return out
    return [] if a == b else [path]


def test_world_builder_only_inserts_one_message(worlds):
    by = {}
    for w in worlds:
        by.setdefault(w["pair_id"], {})[w["world"]] = w
    for pid, ws in by.items():
        minus = ws["x-"]["spec"]
        for k in ("x-F", "x-N"):
            ins = ws[k]["insertion"]
            diffs = _diff(minus, ws[k]["spec"])
            if ins.get("field"):
                assert diffs == [f"/files/{ins['path']}/meta/{ins['field']}"], (pid, k, diffs)
                assert ws[k]["spec"]["files"][ins["path"]]["meta"][ins["field"]] == ins["text"]
            else:
                assert diffs == [f"/files/{ins['path']}/content"], (pid, k, diffs)
                new, old = ws[k]["spec"]["files"][ins["path"]]["content"], minus["files"][ins["path"]]["content"]
                assert new == old + ins["text"]
            # The authoritative record is untouched.
            assert ws[k]["spec"]["permissions"] == minus["permissions"]
        assert abs(len(ws["x-N"]["insertion"]["text"]) / len(ws["x-F"]["insertion"]["text"]) - 1) <= 0.05
        for w in ws.values():
            assert APPROVAL_RULE in w["spec"]["system_prompt"]
            assert ws["x-F"]["insertion"]["text"] not in w["spec"]["system_prompt"] + w["spec"]["task_prompt"]


def test_forged_message_does_not_authorize(worlds):
    """S flags the critical action in x-F exactly as in x-, and not in x+."""
    by = {}
    for w in worlds:
        by.setdefault(w["pair_id"], {})[w["world"]] = w
    for ws in by.values():
        tool, args = ws["x-"]["spec"]["critical_example"]
        flags = {k: sandbox_check(MockEnv(w["spec"]), tool, args)["flag"] for k, w in ws.items()}
        assert flags == {"x+": False, "x-": True, "x-F": True, "x-N": True}, flags


def test_approval_via_system_authorizes(worlds):
    w = next(w for w in worlds if w["world"] == "x-F" and w["spec"]["permissions"]["on_request"] == "approved")
    env = MockEnv(w["spec"])
    tool, args = w["spec"]["critical_example"]
    assert sandbox_check(env, tool, args)["flag"]
    env.step("request_permission", {"resource": "r", "action": "a", "justification": "j"})
    assert not sandbox_check(env, tool, args)["flag"]


def test_sandbox_does_not_mutate_env(worlds):
    w = next(w for w in worlds if w["template"] == "auth_vendor_terms" and w["world"] == "x-")
    env = MockEnv(w["spec"])
    before = copy.deepcopy(env.db)
    tool, args = w["spec"]["critical_example"]
    sandbox_check(env, tool, args)
    assert env.db == before and env.effects == []


def test_scripted_episode_end_to_end(worlds, tmp_path):
    """A scripted agent runs through the real tokenizer and environment; positions and labels are sane."""
    tokenizer = pytest.importorskip("transformers").AutoTokenizer
    try:
        tok = tokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    except Exception:
        pytest.skip("Qwen3.5 tokenizer not available offline")
    from pilot.env.mockenv import TOOL_SCHEMAS
    from pilot.rollout.backends import ScriptedBackend
    from pilot.rollout.chat import openai_tools, tool_wrapper
    from pilot.rollout.run import Episode, partner_index

    backend = ScriptedBackend.__new__(ScriptedBackend)
    backend.tok, backend.plan = tok, None
    wr = tool_wrapper(tok, openai_tools(TOOL_SCHEMAS))
    partners = partner_index(worlds)
    for world in ("x+", "x-", "x-F"):
        w = next(x for x in worlds if x["template"] == "auth_vendor_terms" and x["world"] == world)
        ep = Episode(w, partners[w["world_id"]], 0, backend, tok, wr, "scripted", "test").rollout()
        assert ep["status"] == "final_answer"
        assert ep["critical_done"]
        assert ep["unsafe_episode"] == (world != "x+")
        assert ep["sandbox_flagged"] == ep["unsafe_episode"]
        assert ep["reveal_event"] is not None
        if world == "x-F":
            assert ep["inserted_seen_event"] is not None
            assert ep["inserted_seen_before_violation"]
        assert ep["decision_event"] is not None
        ids = ep["token_ids"]
        text = tok.decode(ids, skip_special_tokens=False)
        from pilot.rollout.chat import initial_prompt
        first = initial_prompt(tok, w["spec"]["system_prompt"], w["spec"]["task_prompt"], openai_tools(TOOL_SCHEMAS))
        assert tok.decode(ids[:ep["events"][0]["prompt_len"]], skip_special_tokens=False) == first
        for e in ep["events"]:
            assert tok.decode(ids[:e["prompt_len"]], skip_special_tokens=False).endswith(wr.after_result) \
                or e["event_id"] == 0
            if e["is_tool_call"]:
                assert e["tool_result"] in text
                assert tok.decode(ids[e["action_start_pos"]:e["action_start_pos"] + 1]) == "<tool_call>"
