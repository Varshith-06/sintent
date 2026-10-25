# Intent is not enough: belief readouts vs. consequence models in agent monitoring

Oct 25, 2026 · @Varshith

## Summary

Monitors that read what an agent intends and monitors that predict what its action will do catch different failures. The project measures that split and builds a combined monitor that covers both.

- **The claim.** Intent monitors, such as probes on activations and chain-of-thought readers, catch knowing harm: the agent believes an action is wrong and does it anyway. They miss unknowing harm, where the agent believes the action is fine but it is not, because of forged authority, a hidden fact or an honest mistake. Consequence monitors, which run the action in a sandbox or predict its effects with a learned model, catch unknowing harm. They miss harm with no immediate visible effect, such as a false report or groundwork for later misuse.
- **The evidence it builds on.** In [The Agent Knew](https://arxiv.org/html/2610.17301) (Oct 2026), a readout of an agent's belief that an action was unauthorized stayed silent on prompt injections signed in the user's name. The agent believed the action was authorized. This project turns that accident into a controlled experiment.
- **The design.** Matched worlds in which what the agent believes and what is actually true are set independently, built on existing agent benchmarks with environment labels. Four cells: knowing harm, unknowing harm, hidden or delayed harm, and benign.
- **The results to aim for.**
  1. A double dissociation: each signal near chance in the cell where the other excels.
  2. A combined monitor that beats both at a fixed false-alarm rate.
  3. How close learned consequence models get to simply running the action in a sandbox.
  4. Whole-episode prevention without blocking legitimate work.
- **What it needs.** About 2,500–5,000 H100-hours, a few thousand dollars of API credit, and 1–2 people for 5–6 months.

Working title: *Intent is not enough: a double dissociation between belief readouts and consequence models in agent monitoring.*

## The idea in plain words

An AI agent acts in the world: it moves files, sends messages, changes permissions. Before an action runs, a monitor can ask two different questions.

- **What does the agent think it is doing?** Read its reasoning, or read its activations, the internal numbers it computes for every word. If it believes the action breaks a rule, that shows up inside. This is an intent signal.
- **What will the action actually do?** Run it in a sandbox copy of the environment, or use a model trained to predict effects, and check the result against the rules. This is a consequence signal.

Each question has a blind spot.

- **Intent misses honest harm.** If someone forges the user's permission, or the agent never saw the fact that makes an action forbidden, the agent believes it is doing the right thing. Its internals look calm, and an intent monitor stays quiet.
- **Consequences miss hidden harm.** A false report, a planted backdoor that fires later, or quietly doing worse work changes nothing harmful right now. A consequence check sees nothing, while the agent's internals may show it knows what it is doing.

The project builds test worlds where these cases come apart on purpose, measures each monitor's blind spot, and shows how to combine the two so neither gap stays open.

## Why this idea

Half of the split already has early evidence. The other half, a controlled test of both halves together, and the combined monitor are still open as of October 2026.

| Work | What it shows | What it leaves open |
| --- | --- | --- |
| [The Agent Knew](https://arxiv.org/html/2610.17301) (Oct 2026) | A readout of the agent's belief that an action is unauthorized forecasts violations and lies, and gating on it prevents violations. It stayed silent on AgentDojo injections signed as the user until recalibrated on 29 tasks. | No consequence monitors; unknowing harm met only by accident |
| [Felt, Not Granted](https://zenodo.org/record/20685264) (Jun 2026, preprint) | On a live file-system agent, a frozen authorization probe allowed all 21 elicited unauthorized deletions, while an external task-grounded check separated them perfectly. | One model, one task family, small sample; the reverse blind spot is untested |
| [Recognition Without Enforcement](https://arxiv.org/pdf/2608.28502) (Aug 2026) | Agents linearly encode where an instruction came from and can spot fabricated authority, yet steering on that direction does not stop execution. | Whether a provenance probe catches forged authority that a belief probe misses |
| [DreamGuard](https://arxiv.org/html/2608.05695), [SeerGuard](https://arxiv.org/pdf/2607.15550) (2026) | World models predict an action's consequences before execution and block risky ones. | No comparison with intent signals; harm without an immediate effect |
| [The Agent Clicks Approve](https://arxiv.org/pdf/2610.15138) (Oct 2026) | A check at the action endpoint against the task's authority stopped consent attacks that prompting could not. | External checks only; hidden harm out of scope |
| [OS-Blind](https://arxiv.org/html/2604.10577) (Apr 2026) | Agents cause harm under entirely benign instructions when the harm comes from the environment. | A source of unknowing-harm tasks; no monitoring study |

The gap: nobody has tested both blind spots in one controlled design across models, or shown how much a combined monitor recovers. That is the paper.

Because half the premise has appeared in small studies, re-check novelty before starting (see "Before you start").

## The core concept

&#91;embedded content: concept · the four cells of the benchmark and which monitor each one tests\]

Intent monitors read the horizontal axis and consequence monitors read the vertical one, so each is blind along the axis it does not read: intent monitors in the top-left cell, consequence monitors in the bottom-right one.

A double dissociation needs both blind spots shown in one design, on the same agents and tasks. Earlier small studies showed only that probes miss unknowing harm, which leaves the other half and the combined monitor untested.

One refinement sits inside the intent side. A belief readout tracks the permission the agent feels, while a provenance probe tracks where an instruction came from. Agents can encode that an instruction's authority is fabricated and still follow it, so a provenance probe may catch part of the top-left cell.

## Research questions and hypotheses

Five questions; the first two together make the double dissociation.

| # | Question | Hypothesis | Falsified if | Why it matters |
| --- | --- | --- | --- | --- |
| RQ1 | Do intent monitors miss unknowing harm? | Belief readouts and CoT monitors are near chance on forged authority, hidden facts and honest mistakes, yet strong on knowing harm. | They flag unknowing harm well above chance. | Shows where reading the agent's mind cannot be trusted. |
| RQ2 | Do consequence monitors miss hidden harm? | Sandbox checks and learned consequence models are near chance on false reports, planted backdoors and deliberate underperformance, where intent readouts succeed. | Consequence monitors catch hidden harm reliably. | Completes the double dissociation. |
| RQ3 | Which internal signal matters? | Provenance probes, which read where an instruction came from, catch forged authority that belief probes miss. | No difference between the two probes. | Tells monitor builders what to read inside the model. |
| RQ4 | Does combining help? | A combined monitor beats each signal at a fixed false-alarm rate, with the gain concentrated in each signal's blind cell, and prevents more harm over whole episodes without blocking authorized work. | The combination matches the best single signal. | Turns the finding into a deployable design. |
| RQ5 | Can a learned model replace the sandbox? | Learned consequence models approach the sandbox on familiar tools at a fraction of the latency but degrade on unseen tools; LLM simulators sit in between. | Learned models fall far short even on familiar tools. | Covers actions that cannot be sandboxed, such as payments and external APIs. |

## Related work

Beyond the papers above, related work falls into six groups.

| Group | Work | Relevance |
| --- | --- | --- |
| Intent and belief probes | [Probing the Misaligned Thinking Process](https://arxiv.org/pdf/2606.24251), [FIBS](https://arxiv.org/pdf/2610.12445), [Goodfire's reward-hacking probes](https://arxiv.org/pdf/2609.19101), [HackTrace](https://arxiv.org/html/2610.03055), [Intent-as-a-Tool](https://arxiv.org/pdf/2608.27348) | Intent-side baselines for knowing harm |
| Limits of probes | [Probes read the situation, not the action](https://arxiv.org/abs/2606.30449), [Why safety probes catch liars but miss fanatics](https://arxiv.org/html/2603.25861) | Prior negative results; the unknowing-harm cell generalizes the "fanatic" case |
| Consequence models and guards | [JANUS](https://arxiv.org/pdf/2607.19913), [Software World Models](https://arxiv.org/pdf/2610.04940), [Text world models for agents](https://arxiv.org/pdf/2606.09032), [Pre-Exec Bench](https://arxiv.org/html/2510.09781v1), [ActGuard](https://arxiv.org/pdf/2609.14987), [SARA](https://arxiv.org/pdf/2608.27146) | Consequence-side baselines and pre-execution checks |
| JEPA-style world models | [What drives success with JEPA world models](https://arxiv.org/pdf/2512.24497v1), [JEPA-WAM](https://arxiv.org/pdf/2608.09381), [WASP](https://www.ijcai.org/proceedings/2026/409) | Your original latent world-model idea; all robotics so far |
| Monitoring methodology | [The Window Is in the Reasoning](https://arxiv.org/html/2610.24048), [From Probe Scores to Alarm Policies](https://arxiv.org/pdf/2610.04575), [SLEIGHT-Bench](https://arxiv.org/html/2605.16626), [Auditing Sabotage Bench](https://arxiv.org/pdf/2604.16286) | Within-state scoring, fixed false-alarm budgets, sabotage vs. honest mistakes |
| Benchmarks to build on | [CAPE-Bench code and data](https://github.com/AryaFayyazi/Honesty), AgentDojo, [τ²-bench](https://arxiv.org/abs/2506.07982), [SHADE-Arena](https://arxiv.org/pdf/2506.15740), OS-Blind | Environments with labels from the environment itself |

## Benchmark design

Every harmful task gets a twin that differs only in what the agent can know or only in what is actually true, so each monitor's blind spot can be measured directly.

| Cell | Agent believes | Immediate effect | How it is built | Example |
| --- | --- | --- | --- | --- |
| Knowing harm | Forbidden | Harmful | The forbidding fact is visible; the agent acts anyway | Exports data after reading "do not export" |
| Unknowing harm: forged authority | Allowed | Harmful | Permission arrives through a forged channel | An injection signed as the user |
| Unknowing harm: hidden fact | Allowed | Harmful | The forbidding fact exists but sits outside the agent's view | The policy lives in a file the agent never opens |
| Unknowing harm: honest mistake | Fine | Harmful | Ambiguity or a tool quirk causes damage | A wildcard deletes the wrong folder |
| Hidden or delayed harm | Wrong | Nothing harmful yet | The harm is a message or a setup step | "Permission was denied" after the action already ran |
| Benign twin | Allowed | Harmless | The authorized version of each task | The same export, approved |

**Where the tasks come from.** Extend [CAPE-Bench](https://github.com/AryaFayyazi/Honesty), which already pairs authorized and unauthorized versions of 24 task templates across eight safety families, with forged-authority, hidden-fact and honest-mistake variants. Add AgentDojo for forged authority, OS-Blind-style tasks for honest mistakes, and a coding sandbox for planted backdoors and reward hacking.

**Belief checks.** An unknowing cell only counts if the agent really did not know. Verify it three ways: a probe that decodes the decisive fact from activations, the agent's reasoning, and a post-hoc question. Drop templates that leak.

**Labels.** Violations and state changes come from the environment log. Reports are judged against that log, with blind human audits. Delayed harm is labeled by running each episode to the end.

**Scale.** About 30 templates × 6 cells × 50 scenarios × twins × 4–6 agent models: roughly 100,000–250,000 episodes.

## Monitors

Three intent signals and four consequence signals score the same proposed actions, then get combined.

**Intent side**

| Signal | Reads | Expected to catch | Expected to miss |
| --- | --- | --- | --- |
| Belief readout (The Agent Knew's recipe) | Residual stream just before the tool call | Knowing harm, lies before a report | Forged authority, hidden facts, honest mistakes |
| Provenance probe | Which source an instruction came from | Forged authority | Hidden facts, honest mistakes |
| Chain-of-thought monitor | The agent's reasoning text | Knowing harm the agent puts into words | Unspoken intent, unknowing harm |

**Consequence side**

| Signal | How it works | Expected to catch | Expected to miss |
| --- | --- | --- | --- |
| Sandbox check | Runs the proposed action in a forked copy of the environment and checks the new state against the rules | Every immediate harm | Hidden or delayed harm; actions that cannot be sandboxed |
| LLM simulator | A strong model predicts the action's effect from context | Broad coverage | Costly; may hallucinate effects |
| Learned consequence model | A fine-tuned small model, or a JEPA-style predictor in embedding space, trained on executed actions | Fast, cheap checks on familiar tools | Unseen tools |
| Static policy rules | Action arguments against written rules | Obvious violations | Anything that depends on context |

**Combining them.** Three designs, compared at the same false-alarm rate:

1. Flag if either side flags, with each threshold set so the total false-alarm rate stays fixed.
2. A small classifier stacked on both sides' scores.
3. A cascade: cheap probes and the learned model screen every action, and the sandbox or LLM simulator runs only on flags.

For prevention, gate every action on the combined signal and ask an approver on repeated flags, the policy The Agent Knew found keeps authorized work.

## Experiments

Eight steps; the first replicates prior work so every later number has a trusted baseline.

| ID | What | H100-hours | If it goes wrong |
| --- | --- | --- | --- |
| E0 | Reproduce CAPE-Bench violations and The Agent Knew's belief readout on two models | \~150 | Readout fails to replicate: use the released code as is, and contact the authors |
| E1 | Build the forged-authority, hidden-fact, honest-mistake and hidden-harm variants; run belief checks | \~200–300 | Variants leak the decisive fact: drop those templates |
| E2 | Episodes at scale: 4–6 open-weight agents, every cell, with twins | \~800–1,500 | A cell is too rare for some models: add templates, or use models known to produce it |
| E3 | Intent monitors on every cell: belief readout, provenance probe, chain-of-thought monitor | \~150–300, plus API credit | CoT monitor too costly: score a stratified subset |
| E4 | Consequence monitors: sandbox check, LLM simulator, learned consequence model | \~800–1,800 | Learned model weak: report the cost–accuracy frontier and recommend the sandbox where it is possible |
| E5 | Double-dissociation test and the three combined monitors | CPU only | No crossover: report which half holds; still a useful map of blind spots |
| E6 | Whole-episode prevention with gating, against authorized twins | \~300–600 | Gating blocks legitimate work: add the approver step |
| E7 | Held-out families and unseen tools; cost and latency per action | \~100–300 | Poor transfer: report it and add brief per-family calibration, as The Agent Knew did for AgentDojo |

Total: about 2,500–5,000 H100-hours.

## Evaluation and statistics

The headline test is a crossover: intent beats consequence on hidden harm, and consequence beats intent on unknowing harm, each by a significant margin.

- **Primary metric.** Recall at a fixed false-alarm rate (5% and 1%) on the benign twins, per cell and per monitor. AUROC is reported alongside, never instead, since high AUROC can hide a useless operating point ([From Probe Scores to Alarm Policies](https://arxiv.org/pdf/2610.04575)).
- **Matched comparisons.** AUROC within (model, template) strata, so no monitor wins by recognizing a template, following The Agent Knew. Where agents are resampled from one state, also score within that state, following The Window Is in the Reasoning.
- **The double-dissociation test.** Both simple effects must point in opposite directions with cluster-bootstrap 95% intervals that exclude zero. One significant half is reported as one half, not as a dissociation.
- **Combination gain.** Recall of each combined monitor minus the best single signal at the same false-alarm rate, with paired bootstrap intervals; prevented episodes compared with exact McNemar tests.
- **Prevention.** Whole-episode violation rate in harmful worlds and task completion in authorized twins, with each gated episode followed several more turns so a postponed violation is not counted as prevented.
- **Cost.** FLOPs and latency per checked action for every monitor.
- **Pre-registration.** Lock hypotheses, thresholds and analyses before fresh scenarios exist, and publish every test, including ones that fail.
- **Splits.** Fit everything on development templates; confirm on fresh scenarios and on held-out safety families.

## Compute and team plan

The project needs two to four weeks of an 8×H100 node spread over five to six months, plus one or two people.

| Component | H100-hours | Notes |
| --- | --- | --- |
| Agent episodes, 4–6 open-weight models | 800–1,500 | Multi-turn runs with mock tools; the largest single cost |
| Activation capture and probes | 150–300 | Probes themselves train in minutes |
| LLM simulator | 300–600 | Or a few thousand dollars of API credit for a hosted model |
| Learned consequence model | 500–1,200 | Fine-tune a 7–8B model on about 2 million executed actions; the JEPA-style variant costs less |
| Prevention runs with gating | 300–600 | Each gated episode followed for 6–12 more turns |
| Belief checks, generalization, ablations | 450–800 | Includes reruns |
| **Total** | **2,500–5,000** |  |

**Money.** At the October 2026 median on-demand price of about $3.49 per H100-hour ([price survey](https://www.amcompute.com/gpu-cloud-prices/h100)), that is roughly $9,000–$17,000, less on reserved or spot capacity, plus $2,000–$8,000 of API credit for hosted monitors and judges.

**Hardware.** Larger agents need a multi-GPU node to run with activations captured. Sandbox checks and environment runs are CPU work, so plan for a machine with many cores.

**People.** One person full time for 5–6 months can do it with an experienced advisor. Two people make it comfortable: one on environments and labels, one on monitors and analysis.

**Where compute could come from.** A university or lab cluster, research credit programs, a fellowship that supplies compute, or collaborating with a group that already runs these agent benchmarks.

## Roadmap

&#91;embedded content: plan · phases not to scale · months counted from when compute arrives\]

Each gate is a stop-and-check point: if a gate fails, fix that phase before spending compute on the next. A workshop paper fits after phase 4, about four months in.

## Risks and mitigations

The biggest risks are reviewers calling the blind spot obvious and the unknowing cells not being truly unknowing.

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| "Of course probes can't see what the agent doesn't know" | High | Lead with the reverse blind spot, the size of each gap on realistic tasks, and how much the combination recovers; include provenance probes that partly close the gap |
| Agents in unknowing cells actually saw the decisive fact | Medium | Belief checks by probe, reasoning and post-hoc question; drop leaky templates |
| Hidden harm is too rare to measure | Medium | Use models and templates known to misreport, as in The Agent Knew; add planted-backdoor coding tasks |
| The learned consequence model never approaches the sandbox | Medium | Report the cost–accuracy frontier; recommend the sandbox wherever actions can be sandboxed |
| The sandbox check is too easy to count as a fair consequence baseline | Medium | Report it as an upper bound, and score the LLM simulator and learned model against it |
| Someone publishes the full dissociation first | Medium | Re-check novelty before starting, then post a preprint as soon as the crossover holds |
| Label noise in report judgments | Medium | Gold labels, blind human audits and agreement statistics, as The Agent Knew did |
| Engineering scope | High | Start from CAPE-Bench's released code; run two environments before adding more |

## Publication strategy and odds

This is a main-track paper with a benchmark attached; aim for a NeurIPS or ICLR main track, with the benchmark as a datasets-track backup.

| Venue | What it needs | My estimate |
| --- | --- | --- |
| A workshop at an A\* conference | E0–E5: the crossover on at least two models, written up in 4 pages | About 65–80% |
| NeurIPS or ICLR main track | All eight experiments, pre-registered tests, released benchmark and code | Around or somewhat above the 25–30% base rate when complete and co-authored; lower as a solo first paper |
| A datasets or evaluations track | The benchmark, labels, belief checks and baselines | Similar to the main track; strongest if the benchmark is easy to reuse |

- **Why the estimate is decent.** The framing is new, the experiment is controlled, and the result is useful whichever way it falls: it tells builders which monitors to combine.
- **What pulls it down.** Part of the premise has appeared in small studies, and reviewers may call one half obvious. The reverse blind spot and the combination have to carry the paper.
- **The bar reviewers will compare against.** The Agent Knew pre-registered its tests, released code and data, and evaluated over whole episodes. Match that.
- **Timing.** Pick the first main-track deadline at least six months after compute arrives, with a workshop deadline around month four for early feedback.

## Before you start

This plan was written in October 2026 for later use; check six things before spending any compute.

- [ ] **Novelty.** Search for the double dissociation itself, follow-ups to [Felt, Not Granted](https://zenodo.org/record/20685264), and papers citing [The Agent Knew](https://arxiv.org/html/2610.17301). If both halves and the combination have appeared, narrow to what is still open, such as learned consequence models against the sandbox, or provenance against belief probes.
- [ ] **Benchmark access.** Confirm [CAPE-Bench's code and data](https://github.com/AryaFayyazi/Honesty) are still available, and check their license for reuse.
- [ ] **Models.** Swap in the strongest open-weight agents available then; the plan only needs open weights and tool use.
- [ ] **Compute.** Secure two to four weeks of an 8×H100-class node, spread over five to six months, plus API credit.
- [ ] **People.** Line up an advisor or co-author who has published on agent monitoring.
- [ ] **Deadlines.** Re-check main-track and workshop dates and work backwards six months.

## Glossary

| Term | Plain meaning |
| --- | --- |
| Intent signal | Evidence about what the agent thinks it is doing, read from its reasoning or activations |
| Consequence signal | Evidence about what an action will actually do, from a sandbox run or a predictive model |
| Belief readout | A probe that reads whether the agent believes an action is allowed |
| Provenance probe | A probe that reads where an instruction came from, such as the user or a web page |
| Knowing harm | The agent believes the action is wrong and does it anyway |
| Unknowing harm | The agent believes the action is fine, but it causes harm |
| Hidden or delayed harm | Harm with no harmful immediate effect, such as a false report or a planted backdoor |
| Matched worlds, twins | Two versions of a task that differ in one fact |
| Sandbox check | Running the action in a throwaway copy of the environment and inspecting the result |
| LLM simulator | A language model asked to predict what an action will do |
| Learned consequence model | A model trained on executed actions to predict their effects quickly |
| JEPA-style predictor | A model that predicts the next state in an embedding space rather than as raw text |
| Double dissociation | Two signals each failing where the other succeeds |
| Combined monitor | One alarm built from both kinds of signal |
| Gate, then ask | Block or steer the first flagged action, then ask an approver on later flags |
| False-alarm rate | The share of legitimate actions a monitor flags |
| Recall | The share of harmful actions a monitor catches |
| Pre-registration | Writing down hypotheses and analyses before collecting the data that tests them |

## Sources

- [The Agent Knew: Internal States Forecast Unauthorized Actions and Stop Them](https://arxiv.org/html/2610.17301) (Fayyazi et al., Oct 2026) and [CAPE-Bench code and data](https://github.com/AryaFayyazi/Honesty)
- [Felt, Not Granted: an internal authorization probe inherits the agent's judgment error](https://zenodo.org/record/20685264) (Jun 2026, preprint)
- [Recognition Without Enforcement](https://arxiv.org/pdf/2608.28502) (Aug 2026)
- [The Window Is in the Reasoning](https://arxiv.org/html/2610.24048) (Jia et al., Oct 2026)
- [DreamGuard](https://arxiv.org/html/2608.05695), [SeerGuard](https://arxiv.org/pdf/2607.15550), [JANUS](https://arxiv.org/pdf/2607.19913) (2026)
- [The Agent Clicks Approve](https://arxiv.org/pdf/2610.15138) (Oct 2026)
- [OS-Blind](https://arxiv.org/html/2604.10577) (Apr 2026)
- [Probing the Misaligned Thinking Process of Language Models](https://arxiv.org/pdf/2606.24251) (ICML 2026)
- [Telling FIBS: probes detect sabotage and unverbalized deception](https://arxiv.org/pdf/2610.12445) (Oct 2026)
- [Monitoring and Discovering Reward Hacking with Internal Representations](https://arxiv.org/pdf/2609.19101) (Goodfire, 2026) and [HackTrace](https://arxiv.org/html/2610.03055) (Oct 2026)
- [Intent-as-a-Tool](https://arxiv.org/pdf/2608.27348) (Aug 2026)
- [Internal-state probes read the situation, not the action](https://arxiv.org/abs/2606.30449) (2026) and [Why safety probes catch liars but miss fanatics](https://arxiv.org/html/2603.25861) (Mar 2026)
- [Software World Models](https://arxiv.org/pdf/2610.04940), [text world models for agents](https://arxiv.org/pdf/2606.09032), [Pre-Exec Bench](https://arxiv.org/html/2510.09781v1), [ActGuard](https://arxiv.org/pdf/2609.14987), [SARA](https://arxiv.org/pdf/2608.27146)
- [What drives success with JEPA world models](https://arxiv.org/pdf/2512.24497v1), [JEPA-WAM](https://arxiv.org/pdf/2608.09381), [WASP](https://www.ijcai.org/proceedings/2026/409)
- [From Probe Scores to Alarm Policies](https://arxiv.org/pdf/2610.04575), [SLEIGHT-Bench](https://arxiv.org/html/2605.16626), [Auditing Sabotage Bench](https://arxiv.org/pdf/2604.16286)
- [τ²-bench](https://arxiv.org/abs/2506.07982), [SHADE-Arena](https://arxiv.org/pdf/2506.15740)
- [H100 cloud prices, October 2026](https://www.amcompute.com/gpu-cloud-prices/h100)
