# Governed Compilation under Grounded Evaluation (GCGE)

The central question: *Does constitutional governance remain effective when agents are genuinely resistant, heterogeneous, and game-theoretically rational — not just RLHF-aligned?*

- **v3** — the four core experiments that establish the GCGE framework on a single seed and a single backbone.
- **v4** — robustness extensions: multi-seed validation with bootstrap CIs, topology robustness, OAT sensitivity, agent-temperature sweep, and homogeneous + heterogeneous cross-backbone generalisation.
- **v4.1** — adds an **R6 N=100 scale spot-check** to v4, replicating the governed-vs.-unconstrained comparison of Experiment 1 at three-fold larger population, with everything else held at the v4 baseline. The four core experiments and R1–R5 remain unchanged.
- **v5** — ECS metric ablations (multiplicative / additive / weighted / no-integrity, plus a graded integrity-penalty sweep), rank-based effect-size geometry with bootstrap confidence intervals, and a governance-layer latency micro-benchmark. v5 reanalyses the existing five-seed v4 logs and runs no new LLM calls.
- **v5.1** — adds a **constraint-threshold sensitivity sweep** and explicit **external governance baselines** (Constitutional-AI-style critique-and-revise, LlamaGuard-style safety-classifier best-of-*N*) on the same kind of synthetic adversarial pools used in the v5 latency benchmark. v5.1 also patches the ECS row of the v5 effect-size panel to use the **mean-of-products** convention (per-timestep ECS averaged over the last 20 steps) so it matches the convention used everywhere else in the paper (abstract, multi-seed table, robustness section); the other four rows are unchanged.

The notebooks share architecture and metric definitions; v4 reuses v3 primitives unchanged and adds the new sweep machinery on top, v4.1 adds R6 on top of v4 with no change to R1–R5, v5 reads only from v4's logged outputs (so it is purely analytical and CPU-only), and v5.1 reuses v5's plumbing and adds two further pure-Python analyses on synthetic adversarial pools (no new LLM calls). All notebooks are independently runnable.

---

## Overview

### The v2 Problem: RLHF Cooperation Bias

In v2.0, LLM agents asked to choose cooperate/defect almost always chose `C`, producing flat dynamics where governance had no measurable effect. Constitutional filtering of influence policies was effectively invisible because the underlying agent decisions were already maximally cooperative regardless of the narrative received.

### The v3 Fix: Hybrid Decision Architecture

v3.0 introduces a three-stage hybrid decision model that decouples the cooperation signal from the LLM's RLHF alignment:

```
base_prob  = f(payoff_history, exploitation_tracking, neighbor_behavior, archetype_bias)
llm_shift  = LLM(narrative, personality, memory)  →  [-0.30, +0.30]
final_prob = clip(base_prob + llm_shift × (1 − resistance), 0.02, 0.98)
action     = sample(Bernoulli(final_prob))
```

The LLM no longer makes a binary C/D decision. It evaluates *how much* the received narrative shifts a game-theoretically computed base probability. This creates genuine defection pressure and makes governance measurably effective.

| Component | v2.0 | v3.0 |
|-----------|------|------|
| Agent action | LLM outputs binary C/D | Game-theoretic base + LLM shift |
| Archetypes | 5 (no bias) | 6 (with `base_bias` + opportunist) |
| Narrative eval | Binary decision prompt | Shift in [-0.30, +0.30] |
| Exploitation tracking | Not tracked | Tracked, penalises base_prob |
| Deployment frequency | Every 10 steps | Every 5 steps |
| Target fraction | 15% | 20% |

### The v4 Extensions: Robustness and Generalisation

v4.0 adds five additional sweeps on top of the v3 baseline, none of which alter the v3 results:

- **R1 — Multi-seed validation** (5 seeds × 3 conditions) with bootstrap 95% CIs (5000 resamples), Cohen's *d*, and Mann–Whitney *U* tests with Bonferroni correction over 6 metrics.
- **R2 — Topology robustness** across Barabási–Albert scale-free, Watts–Strogatz small-world, and Erdős–Rényi random graphs (3 seeds each).
- **R3 — One-at-a-time (OAT) sensitivity** over five GCGE hyperparameters (mean prosociality, target fraction, deploy cadence, candidate pool size, state-noise level), reported as a Normalised Sensitivity Index.
- **R4 — Agent-temperature sweep** over *T* ∈ {0.0, 0.2, 0.5, 0.8} verifying that observed effects are not artefacts of agent-side stochasticity.
- **R5 — Cross-backbone generalisation**: a homogeneous sweep across five backbones (Llama-3.3-70B, Llama-3.1-8B, DeepSeek-V3, Qwen3-235B-A22B, NousResearch Hermes-4-70B) and a heterogeneous sweep that crosses Llama-70B with non-Llama agents and *vice versa*.

### The v4.1 Addition: Scale Spot-Check at N=100

v4.1 adds a new section (R6) on top of v4:

- **R6 — Scale spot-check at N=100.** A single-seed governed-vs.-unconstrained replication of Experiment 1 at three-fold larger population, with everything else held at the v4 baseline (scale-free *m*=3, 50 steps, adversarial *p*<sub>viol</sub>=0.65, seed = 42, Llama-3.3-70B-Instruct on both sides). The qualitative ordering of Experiment 1 is fully preserved at *N*=100; the ECS gap is preserved within multi-seed uncertainty, the integrity collapse of the unconstrained selector is identical, and the autonomy paradox is reproduced exactly. We treat the *N*=100 result as a single-seed existence proof of scaling rather than a full population-size sweep.

### v5: Metric Ablation, Effect-Size Geometry, Latency

v5.0 does not introduce any new agent or governance code, runs no model calls, and operates entirely on the per-component time series (`C`, `A`, `I`, `F`) already logged by v4. The deliverables are:

- **Metric ablation.** The headline ECS gap recomputed under four alternative scoring rules — multiplicative `C·A·I·F`, additive `¼(C+A+I+F)`, weighted-additive `0.40·C + 0.20·A + 0.30·I + 0.10·F`, and integrity-removed `C·A·F` — plus a continuous sweep in which MISLEADING claims are softened from `I=0` to `I=δ` for δ ∈ {0.0, 0.1, 0.2, 0.3, 0.5}. The governed–unconstrained advantage is preserved under additive and weighted forms and decays smoothly but stays positive across the graded sweep; it vanishes only when integrity is dropped entirely. The governed–naive gap is formulation-invariant (≈ −0.004), quantitatively confirming that the hard-constraint floor is the load-bearing component of the framework.
- **Effect-size geometry.** Per-metric raw mean differences with bootstrap (5000 resample) 95 % CIs and rank-based Cliff's δ, alongside the inflated Cohen's *d* / Glass's Δ that the near-deterministic integrity gate produces. Frames the headline as a near-deterministic scoring geometry rather than a conventional behavioural effect.
- **Latency micro-benchmark.** A direct measurement of the governance-selection code (stateless hard rules + one scalar utility, no LLM call) over 2×10⁴ calls on representative six-candidate adversarial pools: mean 0.78 µs (p99 0.88 µs), sustained throughput ≈ 1.3×10⁶ decisions/s/core, and a total filter cost of 7.8 µs for a complete 50-step deployment.

### The v5.1 Addition: Threshold Sweep, External Baselines, and ECS Aggregation Fix

v5.1 adds two further analyses on top of v5 and a small numerical fix to the v5 ECS effect-size row. None of the v5 outputs (ablation panel, latency micro-benchmark, four non-ECS rows of the effect-size panel) change.

- **E — Constraint-threshold sensitivity sweep.** Sweeps the intensity ceiling τ ∈ {0.60, 0.65, …, 0.95} on 2000 synthetic adversarial pools, holding the claims and theme gates fixed, and records candidate rejection rate, selected-policy integrity, selected intensity, and selected manipulation risk. Selected-policy integrity is identically 1.000 and the MISLEADING pass-through rate is identically 0% across the entire sweep; τ controls only the rejection rate (decaying smoothly from 79.8% at τ=0.60 to 26.3% at τ=0.95) and the magnitude of the selected policy's intensity. This closes the question of whether the τ=0.80 operating point used elsewhere is a defensible plateau or a fragile calibration.
- **F — External governance baselines.** Two further governance architectures are reimplemented as deterministic rule-based proxies of their LLM counterparts and run on the same 2000-pool synthetic adversarial workload: a Constitutional-AI-style critique-and-revise loop and a safety-classifier best-of-*N* (LlamaGuard-style) rejection-sampling baseline. All four governance-enabled selectors (governed, naive, critique-revise, safety-classifier) hit the integrity floor of 1.000 and admit 0% MISLEADING claims; they separate cleanly on **secondary** safety dimensions (manipulation risk, BURST timing). The contribution of the GCGE soft scorer is, accordingly, not a marginal gain on the integrity primary — the hard floor saturates that — but a substantial gain on the secondary operational risk surface, which other floor-achieving architectures leave unaddressed.
- **ECS aggregation fix (mean-of-products).** The v5 effect-size panel computed per-seed ECS by multiplying the last-20 means of each component (`mean(C)·mean(A)·mean(I)·mean(F)`). The rest of the paper computes per-seed ECS as the last-20 mean of the per-timestep ECS time series (`mean(C·A·I·F)`). Both are correct but differ by Jensen's inequality in the third decimal. v5.1 reconciles the ECS row of `effect_size_panel.csv` to the time-series convention so it agrees with the abstract, Table `t:multi_seed`, and §4.6 of the paper. The other four rows (cooperation, autonomy, integrity, fairness) are componentwise means and are unaffected.

---

## Agent Architecture

### Personality Archetypes (6)

Each agent is assigned one of six archetypes cycling through the population. Archetypes differ in their `base_bias` (prosocial offset) and `resistance` modifier:

| Archetype | Base bias | Resistance modifier | Behavioural description |
|-----------|-----------|---------------------|------------------------|
| `pragmatist` | 0.00 | — | Responds to economic arguments; sceptical of emotional appeals |
| `idealist` | +0.10 | — | Moved by ethical framing; resistant to purely self-interested appeals |
| `skeptic` | −0.12 | +0.20 | Distrustful of all influence; high baseline resistance |
| `conformist` | 0.00 | −0.15 | Susceptible to social norms; low resistance |
| `free-rider` | −0.15 | — | Exploits cooperators; defection-biased |
| `opportunist` | 0.00 | +0.10 | Context-dependent; resists but may be swayed by economic framing |

### Agent State

Each agent tracks: payoff history (rolling window), exploitation events (cooperated while neighbours defected), times targeted by influence campaigns, times persuaded (shift > 0.02), neighbour cooperation rates, and a smoothed strategy value.

---

## Governance Pipeline

### Influence Compiler

A `COMPILER_MODEL` LLM proposes base policies from a noisy state snapshot. A `CRITIC` then reviews the proposal for polarisation, backlash, equity, and manipulation risk, and may reduce intensity or flag for hard-constraint review.

### Constitutional Constraints

**Hard constraints** (any violation → rejection from the governed pool):
- `claims` must not be `MISLEADING` or `EXAGGERATED`
- `intensity` must not exceed 0.80
- Hub-targeting (`HUBS`) with `intensity > 0.60` is blocked

**Soft penalisation** (applied during utility maximisation):
- High intensity, non-factual claims, and `FEAR`-themed policies all incur penalties
- Among constraint-passing candidates, the policy with the highest penalised utility is selected

### Governance Tags

Delivered narratives are prefixed by a tag that reflects governance mode:
- **Governed**: `[Community-approved message]` — transparent, factual framing
- **Unconstrained** with FEAR/MISLEADING: `[URGENT WARNING]` — pressure framing allowed
- **Naive / Unconstrained (factual)**: no tag

### Three Governance Modes

| Mode | Mechanism |
|------|-----------|
| `governed` | Hard constraints + soft penalisation; factual claims only; intensity capped at 0.80 |
| `naive` | Simple heuristic filtering; blocks only the most extreme violations |
| `unconstrained` | No filtering; highest-utility policy wins regardless of claims or theme |


---

## Experiments

### Core experiments (v3 — `gcge_nebius_v3.ipynb`)

#### Exp 1 — Three-Condition Comparison (Grounded, Adversarial)
`governed` vs `naive` vs `unconstrained` on a scale-free network (30 agents, 50 steps, seed 42) with adversarial candidate injection (70% chance of a violation-carrying candidate per deployment round). This is the core validation of GCGE: governance must remain effective against grounded, resistant agents.

#### Exp 2 — Governance Cost: Grounded vs Idealized
The same governed and unconstrained conditions are run twice — once with the hybrid grounded agent model (Exp 1) and once with the logistic idealized model imported from Papers 2–3. The governance cost (cooperation loss, ECS gain) is computed and compared across regimes, showing whether grounded resistance changes the economics of constitutional filtering.

#### Exp 3 — Adversarial vs Benign (Governed, Grounded)
The governed + grounded condition is run with adversarial injection on and off. This isolates how much of governance's rejection activity is responding to genuinely manipulative candidates versus routine filtering under benign conditions.

#### Exp 4 — Governance × Resistance Interaction Sweep
Five resistance levels (`mean_resistance` ∈ {0.10, 0.25, 0.40, 0.55, 0.70}) are crossed with two governance modes (`governed`, `unconstrained`), yielding 10 simulation runs. The ECS advantage of governance at each resistance level is reported, probing whether constitutional filtering becomes more or less valuable as agents become harder to persuade.

### Robustness extensions (v4 — `gcge_nebius_v4.ipynb`)

The v4 notebook re-runs Experiments 1–4 unchanged, then proceeds to the five new sweeps (R1–R5). All v4 outputs are written under `data_v4/v4/` and are independent of the v3 archive.

#### R1 — Multi-Seed Validation
Three-condition comparison replicated across `seed` ∈ {0, 1, 2, 3, 4} on the scale-free baseline. Reports per-condition bootstrap 95% CIs (5000 resamples) for six metrics, Mann–Whitney *U* (governed vs unconstrained, Bonferroni-corrected for *m* = 6), and Cohen's *d*. The forest plot in `multi_seed/forest_plot.png` summarises the effect-size pattern.

#### R2 — Topology Robustness
Governed vs unconstrained replicated on three topologies (`scale_free`, `small_world`, `erdos_renyi`) with three seeds each. The ECS gap is reported per topology with 95% CIs, demonstrating that the constitutional filter operates above the level of network structure.

#### R3 — OAT Sensitivity
One-at-a-time sweeps over five hyperparameters (`base_prosocial`, `target_frac`, `deploy_every`, `n_candidates`, `state_noise`). The Normalised Sensitivity Index (NSI) for each (parameter × metric) cell is collected in `sensitivity/sensitivity_index.csv` and rendered as a heatmap in `sensitivity/nsi_heatmap.png`.

#### R4 — Agent-Temperature Sweep
Governed and unconstrained conditions replicated at agent-LLM temperatures `T` ∈ {0.0, 0.2, 0.5, 0.8} (compiler temperature held at 0.25). Verifies that the documented effects are not artefacts of agent-side stochasticity; the ECS range across the sweep is below 0.005 in both governance regimes.

#### R5-A — Cross-Backbone Homogeneous
Five backbones run on both compiler and agents: `Llama-3.3-70B-Instruct`, `Llama-3.1-8B-Instruct`, `DeepSeek-V3`, `Qwen3-235B-A22B-Instruct`, `Hermes-4-70B`. Verifies the ECS governance advantage is preserved across model families.

#### R5-B — Cross-Backbone Heterogeneous
Five compiler→agent pairs cross Llama-70B with each non-Llama backbone in both directions. Verifies the constitutional filter does not depend on coordinator and agent population sharing alignment priors.

### Scale spot-check (v4.1 — `gcge_nebius_v4_1.ipynb`)

v4.1 is identical to v4 through R1–R5; it adds a single new section (R6) that runs the only further simulation introduced in the round-2 reviewer response.

#### R6 — Scale Spot-Check at N=100
Single-seed governed-vs.-unconstrained replication of Experiment 1 at *N*=100, with everything else held at the v4 baseline (scale-free *m*=3, 50 time steps, adversarial *p*<sub>viol</sub>=0.65, seed = 42, Llama-3.3-70B-Instruct on both sides). Outputs are written under `data_v5_1/scale_n100/` (one directory per condition) plus a `scale_n100_summary.csv` aggregate and a `scale_n100_comparison.{png,pdf}` overlay against the *N*=30 five-seed means logged by R1.

### Reviewer-response analyses (v5 — `gcge_nebius_v5.ipynb`)

v5 ingests the five-seed v4 components (`components_last20.csv`) and runs three purely analytical passes. No simulations are re-executed.

#### A1 — Metric-formulation ablation
Recomputes the per-condition ECS under four alternative scoring rules and a five-point graded integrity-penalty sweep. Outputs `ablation_ecs_formulations.csv` and the two figures `ablation_formulations.{png,pdf}` (bar chart across formulations) and `ablation_graded_integrity.{png,pdf}` (governance advantage as a continuous function of δ).

#### A2 — Effect-size panel
Reports raw mean differences with bootstrap 95 % CIs, Cliff's δ, Cohen's *d*, and Glass's Δ for cooperation, autonomy, integrity, fairness, and ECS in `effect_size_panel.csv` and `effect_size_panel.{png,pdf}`. Cliff's δ is exactly ±1.0 for ECS, integrity, and autonomy — the honest non-parametric statement of the governed–unconstrained separation.

#### A3 — Governance-layer latency micro-benchmark
Times the constitutional-selection code in isolation (`latency_benchmark.json`). The selector is a stateless set of hard rules plus a single scalar utility evaluation with no LLM call, so its cost is microsecond-order and dominated entirely by the surrounding LLM calls in any realistic deployment.

### Reviewer-response extensions (v5.1 — `gcge_nebius_v5_12.ipynb`)

v5.1 is identical to v5 through A1–A3 with two changes: (i) the ECS row of A2 is reconciled to the mean-of-products convention used everywhere else in the paper (see the v5.1 summary above), and (ii) the notebook adds a fourth, self-contained cell that runs the two further analyses introduced in the round-2 reviewer response. These analyses operate on the same kind of synthetic adversarial candidate pools used in the v5 latency benchmark, so no new LLM calls are required.

#### E — Constraint-threshold sensitivity sweep
Sweeps the intensity ceiling τ ∈ {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95} on 2000 synthetic adversarial pools (six candidates per pool, *p*<sub>viol</sub>=0.65), holding the claims and theme gates fixed. Outputs `threshold_sweep.csv` (one row per τ with rejection rate, selected integrity, selected intensity, selected risk, and MISLEADING pass-through) and `threshold_sweep.{png,pdf}`.

#### F — External governance baselines
Compares the GCGE governed selector against four further selection mechanisms on the same 2000 synthetic adversarial pools: naive (hard-constraint filter only), Constitutional-AI-style critique-and-revise, LlamaGuard-style safety-classifier best-of-*N*, and unconstrained. Outputs `external_baselines.csv` (one row per method with mean integrity, intensity, manipulation risk, MISLEADING/FEAR/BURST selection rates) and `external_baselines.{png,pdf}`.

---

## Repository Structure

```
.
├── README.md
│
│   ── v3 (core experiments, single seed, Llama-3.3-70B) ──
├── gcge_nebius_v3.ipynb            # core notebook (Experiments 1–4)
└── results/                        # v3 outputs (Exp 1–4 + agent analysis)
        ├── governed_adversarial/   # Exp 1 governed
        │   ├── timeseries.csv
        │   ├── policy_log.csv
        │   ├── agents.csv
        │   └── config.json
        ├── naive_adversarial/      # Exp 1 naive
        ├── unconstrained_adversarial/   # Exp 1 unconstrained
        ├── idealized_governed/     # Exp 2 idealized baseline
        ├── idealized_unconstrained/
        ├── governed_benign/        # Exp 3 benign condition
        ├── sweep_governed_R{0.10,0.25,0.40,0.55,0.70}/   # Exp 4
        ├── sweep_unconstrained_R{0.10,0.25,0.40,0.55,0.70}/
        ├── exp1_three_condition.{png,pdf}
        ├── exp1_ecs_decomposition.{png,pdf}
        ├── exp2_grounded_vs_idealized.{png,pdf}
        ├── exp3_adv_vs_benign.{png,pdf}
        ├── exp4_resistance_sweep.{png,pdf}
        ├── agent_analysis.{png,pdf}
        ├── key_message.{png,pdf}
        └── summary.csv
│
│   ── v4 (robustness extensions, R1–R5) ──
├── gcge_nebius_v4.ipynb            # extended notebook (Experiments 1–4 + R1–R5)
└── data_v4/
    ├── v3/                         # v4 re-runs of the v3 experiments (reproducibility)
    │   ├── governed_adversarial/   ...
    │   ├── ...                     (mirrors the layout of /results/)
    │   └── sweep_unconstrained_R0.70/
    │
    └── v4/                         # new sweeps (R1–R5)
        ├── multi_seed/             # R1
        │   ├── governed-s{0..4}/   # per-seed run directories
        │   ├── naive-s{0..4}/
        │   ├── unconstrained-s{0..4}/
        │   ├── multi_seed_summary.csv
        │   ├── bootstrap_ci.csv
        │   ├── mannwhitney_tests.csv
        │   └── forest_plot.{png,pdf}
        │
        ├── topology/               # R2
        │   ├── governed-{scale_free,small_world,erdos_renyi}-s{0..2}/
        │   ├── unconstrained-{scale_free,small_world,erdos_renyi}-s{0..2}/
        │   ├── topology_summary.csv
        │   ├── ecs_gap_by_topology.csv
        │   ├── topology_bars.png
        │   └── ecs_gap_by_topology.png
        │
        ├── sensitivity/            # R3
        │   ├── base_prosocial_{0.30,0.40,0.45,0.55,0.65}/
        │   ├── target_frac_{0.10,0.15,0.20,0.25,0.30}/
        │   ├── deploy_every_{3,5,7,10}/
        │   ├── n_candidates_{4,6,8,10}/
        │   ├── state_noise_{0.05,0.10,0.15,0.20}/
        │   ├── sensitivity_results.csv
        │   ├── sensitivity_index.csv      # NSI per (param × metric)
        │   ├── nsi_heatmap.png
        │   └── sensitivity_analysis.png
        │
        ├── temperature/            # R4
        │   ├── governed-T{0.0,0.2,0.5,0.8}/
        │   ├── unconstrained-T{0.0,0.2,0.5,0.8}/
        │   ├── temperature_summary.csv
        │   └── temperature_robustness.png
        │
        ├── multi_model_homo/       # R5-A
        │   ├── governed-{Llama-70B,Llama-8B,DeepSeek-V3,Qwen3-235B,Hermes-4-70B}/
        │   ├── unconstrained-{Llama-70B,Llama-8B,DeepSeek-V3,Qwen3-235B,Hermes-4-70B}/
        │   ├── multi_model_summary.csv
        │   └── multi_model_bars.{png,pdf}
        │
        └── multi_model_hetero/     # R5-B
            ├── governed-Llama-70B-{DeepSeek-V3,Qwen3-235B,Hermes-4-70B}/
            ├── governed-{DeepSeek-V3,Qwen3-235B}-Llama-70B/
            ├── unconstrained-* (mirror)
            ├── hetero_backbone_summary.csv
            └── hetero_backbone_summary.png
│
│   ── v4.1 (scale spot-check, R6) ──
├── gcge_nebius_v4_1.ipynb          # extended notebook (v4 + R6 N=100 spot-check)
│                                   # New simulation outputs land in data_v5_1/scale_n100/ (see below).
│
│   ── v5 (additional analysis layer, no new model calls) ──
├── gcge_nebius_v5.ipynb            # analysis notebook (A1 ablation + A2 effect sizes + A3 latency)
└── data_v5/
    ├── components_last20.csv            # 5-seed × 3-condition (C, A, I, F) inputs from data_v4/v4/multi_seed/
    ├── ablation_ecs_formulations.csv    # A1: ECS by condition across 4 formulations + graded-δ sweep
    ├── ablation_formulations.{png,pdf}  # A1: bar chart of ECS by condition × formulation
    ├── ablation_graded_integrity.{png,pdf}   # A1: governance gap vs δ
    ├── effect_size_panel.csv            # A2: raw diff, bootstrap CI, Cohen's d, Glass's Δ, Cliff's δ
    │                                    #     (ECS row reconciled to mean-of-products in v5.1)
    ├── effect_size_panel.{png,pdf}      # A2: visual panel
    └── latency_benchmark.json           # A3: n_calls, mean/median/p95/p99 µs, throughput, full-run cost
│
│   ── v5.1 (threshold sweep, external baselines, scale data) ──
├── gcge_nebius_v5_12.ipynb         # extended notebook (v5 + ECS mean-of-products fix + E threshold + F baselines)
└── data_v5_1/
    ├── threshold_sweep.csv              # E: τ vs (rejection rate, sel integrity/intensity/risk, MISL pass-through)
    ├── threshold_sweep.{png,pdf}        # E: plateau + rejection-rate curve
    ├── external_baselines.csv           # F: per-method (integrity, intensity, risk, % MISL/FEAR/BURST)
    ├── external_baselines.{png,pdf}     # F: integrity / risk / BURST comparison
    └── scale_n100/                      # R6 N=100 spot-check (produced by gcge_nebius_v4_1.ipynb)
        ├── governed-N100/
        │   ├── timeseries.csv
        │   ├── policy_log.csv
        │   ├── agents.csv
        │   └── config.json
        ├── unconstrained-N100/
        │   ├── timeseries.csv
        │   ├── policy_log.csv
        │   ├── agents.csv
        │   └── config.json
        ├── scale_n100_summary.csv       # last-20 averages per condition
        └── scale_n100_comparison.{png,pdf}   # N=100 trajectories overlaid on N=30 5-seed means
```

---

## Generated Outputs

The per-run schema is identical across v3, v4, and v4.1 (every run directory under `results/`, `data_v4/v3/`, any v4 sweep, and the v4.1 R6 directories under `data_v5_1/scale_n100/` contains the same four files).

### Per-Run Files

**`timeseries.csv`** — one row per simulation step:

| Column | Description |
|--------|-------------|
| `t` | Simulation step |
| `coop_rate` | Fraction of agents that cooperated this step |
| `ecs` | Ethical Cooperation Score (multiplicative composite) |
| `autonomy_retention` | 1 − (persuasion_rate × 0.35 + backlash_rate × 0.15) |
| `epistemic_integrity` | 1.0 for FACTUAL claims, 0.2 for EXAGGERATED, 0.0 for MISLEADING |
| `subgroup_fairness` | 1 − \|hub cooperation − periphery cooperation\| |
| `avg_payoff` | Mean PD payoff across all agents |
| `payoff_gini` | Gini coefficient of per-agent payoffs |
| `persuasion_rate` | Fraction of targeted agents with effective shift > 0.02 |
| `avg_shift` | Mean effective LLM cooperation shift among targeted agents |
| `backlash_rate` | Fraction of targeted agents with negative shift (narrative backfire) |
| `strategy_mean` | Mean smoothed strategy value (0 = full defection, 1 = full cooperation) |
| `strategy_std` | Standard deviation of smoothed strategies (population heterogeneity) |
| `n_targeted` | Number of agents targeted this step |
| `governance` | Governance mode label |
| `label` | Run label |

**`policy_log.csv`** — one row per influence deployment (every 5 steps):

| Column | Description |
|--------|-------------|
| `t` | Deployment step |
| `target_mode` | Targeting strategy: HUBS, BRIDGES, PERIPHERY, or RANDOM |
| `theme` | Narrative theme: MORAL, ECONOMIC, IDENTITY, HYBRID, or FEAR |
| `intensity` | Policy intensity [0, 1] |
| `timing` | BURST or PERIODIC |
| `claims` | Factual quality: FACTUAL, EXAGGERATED, or MISLEADING |
| `confidence` | Compiler confidence in this policy [0, 1] |
| `reasoning` | LLM-generated brief rationale |
| `n_targets` | Number of agents targeted in this deployment |
| `governance` | Governance mode |
| `n_rejected` | Candidates rejected before this policy was selected |

**`agents.csv`** — one row per agent, post-simulation:

| Column | Description |
|--------|-------------|
| `idx` | Agent index |
| `archetype` | Personality archetype |
| `prosocial_init` | Initial prosocial propensity (drawn at creation) |
| `base_bias` | Archetype-specific cooperation bias |
| `resistance` | Resistance to narrative influence (high = harder to persuade) |
| `final_strategy` | Smoothed final strategy value |
| `cumulative_payoff` | Total PD payoff accumulated over the run |
| `times_targeted` | How many steps this agent was in the target set |
| `times_persuaded` | How many times effective shift exceeded threshold |
| `times_exploited` | How many times agent cooperated while neighbours defected |
| `degree` | Network degree (number of neighbours) |

**`config.json`** — full run configuration plus graph statistics and LLM call counts (compiler and agent roles tracked separately).

### v4 Aggregate Files

In addition to per-run directories, each v4 sweep produces aggregate CSV/PNG files at the sweep root.

**`multi_seed/multi_seed_summary.csv`** — one row per (condition × seed) summarising last-20 averages of every metric.

**`multi_seed/bootstrap_ci.csv`** — per-metric mean and 95% CI lower/upper bounds for each of the three governance conditions (5000 resamples).

**`multi_seed/mannwhitney_tests.csv`** — *U* statistic, raw *p*, Bonferroni-corrected *p* (over 6 metrics), Cohen's *d*, and significance flag for both `governed_vs_unconstrained` and `governed_vs_naive` comparisons.

**`topology/topology_summary.csv`** — one row per (topology × condition × seed) with last-20 averages.

**`topology/ecs_gap_by_topology.csv`** — per-seed ECS gap (governed − unconstrained) by topology; consumed by `ecs_gap_by_topology.png`.

**`sensitivity/sensitivity_results.csv`** — one row per swept value across the five OAT parameters with all metric averages.

**`sensitivity/sensitivity_index.csv`** — NSI matrix: rows are parameters, columns are metrics. The integrity column is identically zero — a structural consequence of the multiplicative gate in the ECS functional.

**`temperature/temperature_summary.csv`** — one row per (governance × temperature) with last-20 averages.

**`multi_model_homo/multi_model_summary.csv`** — one row per (backbone × condition) for the homogeneous sweep.

**`multi_model_hetero/hetero_backbone_summary.csv`** — one row per (compiler→agent pair × condition) for the heterogeneous sweep.

### v5 Analysis Files

The v5 notebook reads the five-seed components logged by R1 and writes its outputs into `data_v5/`. No per-run subdirectories exist — v5 produces tables and figures, not simulation traces.

**`components_last20.csv`** — input table: one row per (condition × seed) with last-20 averages of cooperation `C`, autonomy `A`, integrity `I`, and fairness `F`. Sourced from `data_v4/v4/multi_seed/*/timeseries.csv`; loaded by v5 and held fixed across all ablation passes. v5.1 additionally records a per-seed last-20 mean of the per-timestep ECS time series in a new `ECS_ts` column so that the ECS effect-size row can be computed under the same convention as the rest of the paper.

**`ablation_ecs_formulations.csv`** — five rows (`governed`, `naive`, `unconstrained`, `gap_gov_unc`, `gap_gov_naive`) by nine columns covering the four formulations (`mult`, `add`, `wadd`, `noI`) and the graded-δ sweep at δ ∈ {0.0, 0.1, 0.2, 0.3, 0.5}. The `gap_gov_naive` row is ≈ −0.004 across every formulation, confirming formulation-invariance of the load-bearing component. All columns use the product-of-means convention so the four formulations are like-for-like comparable.

**`ablation_formulations.{png,pdf}`** — grouped bar chart: governed / naive / unconstrained ECS under the four formulations.

**`ablation_graded_integrity.{png,pdf}`** — line plot of the governance gap Δ ECS as a continuous function of the integrity penalty δ assigned to MISLEADING claims.

**`effect_size_panel.csv`** — one row per metric (`C`, `A`, `I`, `F`, `ECS`) with `mean_gov`, `mean_unc`, `raw_diff`, bootstrap `ci_lo` / `ci_hi`, `cohen_d`, `glass_delta`, and `cliffs_delta`. *Note (v5.1):* the ECS row uses the mean-of-products convention (per-timestep ECS averaged over last 20 steps) so that it agrees with the abstract, Table `t:multi_seed`, and §4.6 of the paper. The four componentwise rows (`C`, `A`, `I`, `F`) are unaffected and unchanged from v5.

**`effect_size_panel.{png,pdf}`** — visual rendering of the effect-size panel.

**`latency_benchmark.json`** — single JSON object: `n_calls`, `mean_us`, `median_us`, `p95_us`, `p99_us`, `throughput_per_s`, and `full_run_10calls_us`. Measured on the exact governance-selection code, in isolation from the LLM pipeline.

### v5.1 Analysis Files (`data_v5_1/`)

The v5.1 outputs are placed in a separate `data_v5_1/` directory so the round-2 deliverables are easy to inspect at a glance without mixing them with the original v5 files.

**`threshold_sweep.csv`** — one row per τ ∈ {0.60, 0.65, …, 0.95}, columns `tau`, `rej_rate`, `sel_integrity`, `sel_intensity`, `sel_risk`, `misleading_pass`. Selected integrity is identically 1.000 and MISLEADING pass-through identically 0.0 across the entire sweep; the rejection rate decays smoothly from 79.8% at τ=0.60 to 26.3% at τ=0.95.

**`threshold_sweep.{png,pdf}`** — two-panel figure. Left: selected-policy integrity and MISLEADING pass-through (both flat at the safety floor across the sweep). Right: candidate rejection rate and selected intensity (the only quantities that respond to τ). The paper operating range τ ∈ [0.75, 0.85] is shaded.

**`external_baselines.csv`** — one row per method (`governed (this work)`, `naive (hard filter)`, `critique-revise (CAI-style)`, `safety-classifier (LG-style)`, `unconstrained`), columns `mean_integrity`, `mean_intensity`, `mean_risk`, `pct_misleading`, `pct_fear`, `pct_burst`, `n_pools`. All four governance-enabled methods hit `mean_integrity = 1.000` and `pct_misleading = 0.0`; they separate cleanly on secondary safety dimensions.

**`external_baselines.{png,pdf}`** — three-panel comparison figure. Left: integrity floor (1.000 for the four governance-enabled selectors; 0.171 for unconstrained). Centre: mean manipulation risk. Right: % BURST timing in selected policy. The safety-classifier in particular admits 65% BURST policies that the GCGE soft scorer rejects.

**`scale_n100/scale_n100_summary.csv`** — two rows (`governed`, `unconstrained`) with last-20 averages of cooperation, ECS, autonomy, integrity, fairness, and backlash at *N*=100. Generated by `gcge_nebius_v4_1.ipynb` R6.

**`scale_n100/scale_n100_comparison.{png,pdf}`** — two-panel overlay. Solid lines: *N*=100 trajectories for governed (green) and unconstrained (red). Dashed lines: *N*=30 five-seed means from `data_v4/v4/multi_seed/`. The *N*=100 curves track the *N*=30 baseline almost exactly on both cooperation and ECS.

**`scale_n100/{governed,unconstrained}-N100/`** — per-run directories (same four-file schema as v3/v4) for the two *N*=100 conditions, allowing the spot-check to be re-aggregated or re-plotted from raw traces if needed.

### Root-Level Figures (v3 only — v4 equivalents are inside their sweep directories)

All figures are saved as PNG at 300 dpi.

**`exp1_three_condition.png`** — 3-panel horizontal figure (16 × 4.5 in). Panels: cooperation rate over time, ECS over time, and autonomy retention over time. Three lines per panel: Governed (green), Naive (amber), Unconstrained (red). Shows whether governance imposes a cooperation cost while preserving ECS.

**`exp1_ecs_decomposition.png`** — 4-panel horizontal figure (18 × 4 in). Each panel tracks one ECS component over time: cooperation, autonomy, integrity, and fairness. Reveals which component governance protects most and which the unconstrained condition sacrifices first.

**`exp2_grounded_vs_idealized.png`** — 2-panel figure (14 × 5 in). Left panel: cooperation rate for grounded vs idealized agents, both governed and unconstrained (solid = grounded, dashed = idealized). Right panel: same for ECS. Measures the governance cost gap between the two agent regimes.

**`exp3_adv_vs_benign.png`** — 2-panel figure (12 × 4.5 in). Cooperation and ECS trajectories for the governed + grounded condition under adversarial (red) vs benign (green) influence injection.

**`exp4_resistance_sweep.png`** — 2-panel figure (14 × 5 in). X-axis: mean resistance level (0.10 → 0.70). Left panel: final cooperation averaged over last 20 steps. Right panel: final ECS. Two lines per panel: Governed vs Unconstrained. Shows how governance advantage evolves as population resistance increases.

**`agent_analysis.png`** — 3-panel figure (15 × 4.5 in) drawn from the Exp 1 governed run's `agents.csv`:
- *Left*: scatter of final strategy vs resistance, coloured by archetype
- *Centre*: horizontal bar chart of mean times persuaded per archetype
- *Right*: scatter of cumulative payoff vs network degree, coloured by final strategy

**`key_message.png`** — Single-panel ECS over time for all three Exp 1 conditions (16 × 6 in), with an annotated callout: *"Governance preserves ethics even with resistant agents"*. This is the paper's headline figure.

**`summary.csv`** — Consolidated metrics across all v3 experiments. Columns: `label`, `governance`, `adversarial`, `coop_final`, `ecs_final`, `autonomy_final`, `integrity_final`, `fairness_final`, `avg_shift_final`, `backlash_rate_final`, `rejection_rate`. Metrics are averaged over the last 20 steps of each run.

---

## Key Metric: ECS

**Ethical Cooperation Score** is a multiplicative composite defined consistently across the paper series:

```
ECS = cooperation × autonomy × integrity × fairness
```

The multiplicative form ensures that a collapse in any single component — even if raw cooperation is maintained — registers as a system-level failure. This is the primary differentiator between the governance conditions: the unconstrained mode may match or exceed the governed mode on raw cooperation, but cannot match it on ECS because integrity and autonomy erode under manipulative policies.

### Aggregation conventions

Two arithmetically distinct ways of summarising per-seed ECS exist, both internally valid:

- **Mean-of-products** (the time-series convention): for each seed, compute ECS at every time step as `C_t · A_t · I_t · F_t`, then take the last-20 mean. This is what the paper reports in the abstract, in Table `t:multi_seed` (multi-seed), in §4.6, in Table `t:effect` (effect-size panel), and in the *N*=100 spot-check. It is also what `data_v4/v4/multi_seed/*/timeseries.csv` logs in the `ecs` column. From v5.1 onward, the ECS row of `data_v5/effect_size_panel.csv` is computed under this convention so the panel agrees with the rest of the paper.
- **Product-of-means** (the formulation-ablation convention): for each seed, take the last-20 mean of each component separately, then form `mean(C) · mean(A) · mean(I) · mean(F)`. This is what `data_v5/ablation_ecs_formulations.csv` uses, so that the four alternative ECS formulations (multiplicative, additive, weighted-additive, integrity-removed) and the graded-δ sweep are like-for-like comparable as functions of the same per-seed component means.

By Jensen's inequality applied to a product of correlated bounded variables, the two conventions agree to within ≈ 0.001 on this data. The paper reports `ΔECS = +0.152` (mean-of-products) in the multi-seed table and the effect-size panel, and `+0.153` (product-of-means) in the formulation-ablation table; both are correct under their respective conventions.

---

## Setup

### Requirements

```bash
pip install openai networkx tqdm matplotlib pandas numpy scipy
```

`scipy` is required for the v4 and v4.1 notebooks (bootstrap and Mann–Whitney *U*); the v3 notebook does not depend on it. v5 and v5.1 use only `pandas`, `numpy`, `matplotlib`, and `scipy`, and make no LLM calls, so an API key is not required for the analysis layers.

### API Key

```bash
export NEBIUS_API_KEY="your_key_here"
```

Or set it via Google Colab Secrets (`NEBIUS_API_KEY`). The simulation notebooks (v3, v4, v4.1) use two separate `NebiusLLM` instances with independent call tracking:

- **Compiler** (`role="compiler"`, temp = 0.25, max_tokens = 400) — policy generation
- **Agent** (`role="agent"`, temp = 0.30, max_tokens = 250) — narrative shift evaluation

Default model for both: `meta-llama/Llama-3.3-70B-Instruct` via `https://api.studio.nebius.com/v1/`.

R5 (cross-backbone) additionally uses, via the same Nebius endpoint:
`meta-llama/Llama-3.1-8B-Instruct`,
`deepseek-ai/DeepSeek-V3`,
`Qwen/Qwen3-235B-A22B-Instruct-2507`,
`NousResearch/Hermes-4-70B`.

### Running

Execute all cells in order.

- `gcge_nebius_v3.ipynb` runs the four core experiments sequentially from the orchestration cell. Outputs are written to `/content/outputs/paper5_v3/` (Colab) or the working directory.
- `gcge_nebius_v4.ipynb` first re-runs Experiments 1–4 (writing to `paper5_v4/v3/`), then proceeds to R1–R5 (writing to `paper5_v4/v4/<sweep_name>/`). The v4 notebook is *additive* with respect to v3: if you only need the core experiments, run v3.
- `gcge_nebius_v4_1.ipynb` is identical to v4 through R1–R5 and adds an R6 *N*=100 scale spot-check. Outputs of R6 are written to `data_v5_1/scale_n100/`. The R6 cell can be executed in isolation after R1 has populated `data_v4/v4/multi_seed/` (R6 reads the multi-seed means to overlay them on the *N*=100 trajectories).
- `gcge_nebius_v5.ipynb` is an analysis layer. It reads `data_v4/v4/multi_seed/*/timeseries.csv` for the per-component last-20 averages and writes its outputs to `data_v5/`. Runs in seconds on CPU; no API key required.
- `gcge_nebius_v5_12.ipynb` is the v5.1 extended analysis layer. Cells 1–2 are the patched v5 (mean-of-products ECS row); cell 4 is the round-2 extension that runs the threshold sweep (E) and the external-baselines comparison (F) on synthetic adversarial pools, writing to `data_v5_1/`. By default the extended cell respects the `GCGE_V5_OUT` environment variable; set it to `data_v5_1` to land outputs in the round-2 directory:

  ```bash
  GCGE_V5_OUT=data_v5_1 jupyter nbconvert --execute --inplace gcge_nebius_v5_12.ipynb
  ```

---

## Configuration

Key parameters in `GCGEConfig` (identical between v3 and v4):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 30 | Population size |
| `topo` | `scale_free` | Network topology (`small_world`, `scale_free`, `erdos_renyi`) |
| `steps` | 50 | Simulation steps |
| `governance_mode` | `governed` | `governed`, `naive`, or `unconstrained` |
| `adversarial` | `True` | Inject adversarial policy candidates (70% violation probability) |
| `n_candidates` | 6 | Candidates generated per deployment round |
| `deploy_every` | 5 | Steps between influence deployments |
| `target_frac` | 0.20 | Fraction of agents targeted per deployment |
| `resistance_mean` | 0.30 | Mean agent resistance to narrative influence |
| `resistance_std` | 0.15 | Standard deviation of resistance |
| `base_prosocial` | 0.45 | Mean initial prosocial propensity |
| `state_noise` | 0.10 | Gaussian noise on state snapshots fed to the compiler LLM |
| `seed` | 42 | Random seed for reproducibility |

The v4 sweeps override these defaults in tightly scoped ways — for instance, R2 varies only `topo` and `seed`, R3 varies one parameter at a time, R5 varies the LLM backbone strings only, and R6 (v4.1) varies only `n_agents` (30 → 100) — keeping all unmentioned parameters at the table defaults so each sweep result remains directly comparable to the v3 baseline.

The v5.1 synthetic-pool analyses (E threshold sweep, F external baselines) operate on candidate pools that mirror the v4 stress-test generator (`make_stress`, six candidates per pool, violation probability `pv = 0.65`). The hard-constraint thresholds these analyses sweep against — `max_intensity`, `forbid_claims`, `forbid_themes`, and the soft-scorer penalties — are exactly the values used in the simulation notebooks (`max_intensity = 0.80`, forbid `{EXAGGERATED, MISLEADING}` claims and `{FEAR}` themes).

---

## Citation

If you use this code or build on this work, please cite the associated paper (reference to be updated upon publication).
