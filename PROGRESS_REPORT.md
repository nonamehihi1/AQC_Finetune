# Progress Report — AQC: Adaptive Q-Chunking

**Last Updated**: 2026-08-12

---

## Summary

We are extending Q-Chunking (QC) with **Adaptive Q-Chunking (AQC)** — a method that selects chunk sizes dynamically based on state, using **Learned Variance Normalization (LVN)** to replace noisy sample-based Z-score computation.

---

## Completed Work

### 1. Literature Review & Problem Identification
- [x] Read and analyzed the Q-Chunking paper (Li et al., NeurIPS 2025)
- [x] Identified the fixed-chunk-size limitation as the primary area for improvement
- [x] Documented 4 potential improvement directions (see `AQC_proposed_improvements.md`)
- [x] Performed feasibility analysis of all 4 proposals → selected **LVN** as highest-priority

### 2. AQC Agent Implementation (`agents/aqc.py`)
- [x] Implemented multi-scale critic architecture: separate $Q^k$, $V^k$, $M^k$ networks for each $k \in \{1, 3, 5\}$
- [x] Implemented critic loss function with:
  - $Q^h$ loss (standard FQL on full horizon)
  - $V^h$ loss (expectile regression)
  - Per-scale $Q^k$, $V^k$, $M^k$ losses with proper bootstrapping via $V^h(s_{t+k})$
- [x] Implemented `sample_actions_adaptive()`: generates candidate action sequences, evaluates normalized advantage $\tilde{z}^k = A^k / \sqrt{M^k}$ across all chunk sizes, jointly selects best $(k^*, a^*)$
- [x] Fixed dimension mismatch bug in `sample_actions_adaptive()` for broadcasting V^k and M^k tensors

### 3. Evaluation Support (`evaluation.py`)
- [x] Modified evaluation loop to auto-detect AQC agent (via `hasattr(agent, 'sample_actions_adaptive')`)
- [x] Evaluation correctly truncates action chunks to $k^*$ selected steps

### 4. Quick Smoke Test
- [x] Verified AQC training runs without crashes (10 offline + 15 online steps)
- [x] Confirmed loss values (Q, V, M losses) are reasonable and decreasing

---

## In Progress

### 5. Full-Scale Experiments (Kaggle GPU T4x2)

**Experiment plan:**

| # | Method | Environment | Steps | Actor Type | Status |
|---|--------|-------------|-------|-----------|--------|
| 1 | ACFQL baseline (h=5) | `cube-triple-play-singletask-task4-v0` | 1M offline + 1M online | `distill-ddpg` | ⏳ Queued |
| 2 | AQC-LVN (k∈{1,3,5}) | `cube-triple-play-singletask-task4-v0` | 1M offline + 1M online | `distill-ddpg` | ⏳ Queued |

**Estimated time**: ~8-11 hours per experiment on Kaggle GPU T4.

**Metrics to compare**: 
- Success rate (primary)
- Training curves (success rate vs steps)
- Distribution of selected $k^*$ values across states
- Training speed (iterations/second)

---

## Next Steps

### Short-term (This week)
- [ ] Run Experiment 1 (ACFQL baseline) on Kaggle
- [ ] Run Experiment 2 (AQC-LVN) on Kaggle
- [ ] Compare results and generate plots
- [ ] If AQC-LVN underperforms: ablation study to diagnose (M^k learning, z-score calibration, etc.)

### Medium-term (If LVN works)
- [ ] Run on additional environments (task2, task3) and seeds (3-5 seeds for statistical significance)
- [ ] Ablation: compare LVN vs original Z-score vs Running EMA normalization
- [ ] Ablation: vary $N$ (number of candidate samples) — expect LVN to be less sensitive to $N$ than Z-score
- [ ] Analyze k* distribution: does AQC actually choose different k for different states?

### Exploration (If time permits)
- [ ] Implement simplified LVN variant using Running EMA (no extra network)
- [ ] Experiment with Unified Multi-Scale Critic (Proposal 2) — horizon embedding approach
- [ ] Investigate surprise-based re-planning during chunk execution

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| AQC-LVN underperforms baseline | Medium | LVN has more networks to train → may need longer training or hyperparameter tuning. Start with same hyperparams as ACFQL. |
| GPU time insufficient | Low | Using `distill-ddpg` actor reduces training time by ~4x. Each experiment fits within Kaggle's 12h session limit. |
| M^k network doesn't converge well | Medium | Monitor M^k loss during training. If unstable, try Running EMA alternative. |
| Dimension/shape bugs in AQC | Low | Already fixed one major bug. Quick smoke test passes. |

---

## Files Changed (vs Original QC Codebase)

| File | Change Type | Description |
|------|-----------|-------------|
| `agents/aqc.py` | **NEW** | AQC agent: multi-scale critics, LVN, adaptive chunk selection |
| `agents/__init__.py` | MODIFIED | Registered AQC agent in agent registry |
| `evaluation.py` | MODIFIED | Added adaptive chunk support in eval loop |
| `main.py` | MODIFIED | Added `--chunk_sizes` flag |
| `AQC_proposed_improvements.md` | **NEW** | Detailed proposal document (4 improvement ideas) |
| `PROGRESS_REPORT.md` | **NEW** | This file |
| `README.md` | MODIFIED | Added general idea and run instructions |
