# Governed Compilation under Grounded Evaluation (GCGE)

The central question: *Does constitutional governance remain effective when agents are genuinely resistant, heterogeneous, and game-theoretically rational — not just RLHF-aligned?*

- **v3** — the four core experiments (Sections 4.1–4.5 of the paper) that establish the GCGE framework on a single seed and a single backbone.
- **v4** — robustness extensions (Sections 4.6–4.7 of the paper): multi-seed validation with bootstrap CIs, topology robustness, OAT sensitivity, agent-temperature sweep, and homogeneous + heterogeneous cross-backbone generalisation.

The two notebooks share architecture and metric definitions; v4 reuses v3 primitives unchanged and adds the new sweep machinery on top. Both are independently runnable.

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
```

---

## Generated Outputs

The per-run schema is identical across v3 and v4 (every run directory under `results/`, `data_v4/v3/`, and any v4 sweep contains the same four files).

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

---

## Setup

### Requirements

```bash
pip install openai networkx tqdm matplotlib pandas numpy scipy
```

`scipy` is required for the v4 notebook (bootstrap and Mann–Whitney *U*); the v3 notebook does not depend on it.

### API Key

```bash
export NEBIUS_API_KEY="your_key_here"
```

Or set it via Google Colab Secrets (`NEBIUS_API_KEY`). Both notebooks use two separate `NebiusLLM` instances with independent call tracking:

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

The v4 sweeps override these defaults in tightly scoped ways — for instance, R2 varies only `topo` and `seed`, R3 varies one parameter at a time, R5 varies the LLM backbone strings only — keeping all unmentioned parameters at the table defaults so each sweep result remains directly comparable to the v3 baseline.

---

## Citation

If you use this code or build on this work, please cite the associated paper (reference to be updated upon publication).
