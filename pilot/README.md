# Intent vs. consequence monitors: pilot

Does an intent readout stay quiet on forged-authority violations that a sandbox check catches? This repo
runs that test on one small open-weight agent (Qwen3.5-9B) on a laptop GPU (RTX 4060, 8 GB). It builds
on CAPE-Bench from [The Agent Knew](https://arxiv.org/abs/2610.17301) (Fayyazi, Kamal, Pedram; USC).

The plan, with frozen hypotheses and outcome criteria: [`plan/PLAN.md`](plan/PLAN.md).

## What is here

```
pilot/
  env/mockenv.py, env/sql.py   deterministic mock environment rebuilt from the CC BY 4.0 scenario specs
  env/rules.py                 rule-based checker over authoritative state (ground truth and monitor S)
  env/replay_released.py       Gate 1: replay released CAPE-Bench episodes, compare labels
  worlds/build_worlds.py       x+, x-, x-F (forged approval), x-N (nuisance) per scenario
  rollout/run.py               agent loop, append-only token transcripts, events_json-style logs
  rollout/backends.py          llama.cpp / vLLM server, transformers 4-bit, scripted agent
  rollout/chat.py              chat-template pieces and tool-call parsing
  replay/activations.py        teacher-forced pass; hooks on 5 depths at the read positions only
  monitors/sandbox.py          S: fork the env, run the call, check authoritative state
  monitors/readout.py          R1 decision-state readout, R2 pre-action gate (shrinkage LDA)
  belief/check.py              forked post-hoc question + reasoning scan; blind hand-label sheet
  analysis/score.py            recall at fixed false alarms, AUROC, cluster bootstrap, outcome
  analysis/figure.py           headline figure
tests/test_pilot.py            Gate 1 replay, world-builder diff, forged-approval and pipeline tests
```

## Why the environment is rebuilt

The CAPE repository has no LICENSE file, so its code is not reused. The mock tools and the safety checker
were reimplemented from the CC BY 4.0 dataset (scenario specs and the tool outputs visible in the
released episodes). **Gate 1 passes:** replaying every logged tool call from the released `core24`
episodes (4,000 episodes, 30,090 calls, all 24 templates, Qwen3.8-27B), our checker and the sandbox
monitor agree with every `unsafe_action` and `critical` label (`results/gate1_replay.json`).

## Setup (Windows, native, no WSL)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m pytest tests -q          # downloads CAPE-Bench parquet files on first run
```

Python 3.14 is not used: its `tokenizers` wheel corrupts the heap on Windows.

## Run

```powershell
powershell -File start-server.ps1                      # terminal 1: llama.cpp, 4 slots x 8k context
powershell -File run_pilot.ps1 -Stage rollout          # terminal 2: 960 episodes, resumable
powershell -File run_pilot.ps1 -Stage belief
powershell -File run_pilot.ps1 -Stage replay           # stops the server; transformers needs the GPU
powershell -File run_pilot.ps1 -Stage analyze          # results\pilot_main\report.md + headline_figure.png
```

Single steps, for smoke tests:

```powershell
.venv\Scripts\python.exe -m pilot.env.replay_released --all-templates        # Gate 1
.venv\Scripts\python.exe -m pilot.worlds.build_worlds                        # data\worlds.jsonl
.venv\Scripts\python.exe -m pilot.rollout.run --backend server --run smoke_9b --per-template 1 --seeds 0 --workers 4
```

Measured speeds on the RTX 4060 laptop GPU: llama.cpp Q4_K_M about 28 tokens/s per stream, with four
parallel streams; transformers + bitsandbytes 4-bit about 3.5 tokens/s (usable for replay, which is one
forward pass per episode, but too slow for generation).

## Limits to state in every write-up

- S is an upper bound for consequence monitors: it reads the authoritative state directly.
- One small model, six templates, mock tools.
- Generation (llama.cpp Q4_K_M) and replay (transformers NF4) use different quantizations of the same
  weights; the replay-equivalence gap is measured and reported (`replay_equivalence.json`).

## Data

CAPE-Bench (CC BY 4.0): https://huggingface.co/datasets/Afayyazi/CAPE-Bench. Episode logs and
activations go to `data/` (git-ignored except the world specs).
