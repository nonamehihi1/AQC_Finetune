<div align="center">

# AQC: Adaptive Q-Chunking with Learned Variance Normalization

### Extending [Q-Chunking (NeurIPS 2025)](https://arxiv.org/abs/2507.07969) with adaptive chunk selection

</div>

---

## General Idea

### Background: Q-Chunking (QC)

Q-Chunking (Li et al., NeurIPS 2025) proposes **action chunking for RL**: instead of selecting one action per timestep, the agent commits to a sequence of $h$ actions (a "chunk"), enabling temporally coherent exploration and leveraging offline demonstration data more effectively. Combined with Flow Q-Learning (FQL), QC achieves state-of-the-art results on dexterous manipulation tasks in OGBench.

**Key limitation**: QC uses a **fixed chunk size** $h$ for all states. In practice, some states benefit from long chunks (e.g., reaching motions in free space), while others require short chunks (e.g., precise contact with an object). A fixed $h$ is suboptimal — too long causes sluggish reactions near contacts, too short wastes the benefits of temporal abstraction.

### Our Proposal: Adaptive Q-Chunking (AQC)

We extend QC with **state-dependent adaptive chunk selection**. The core idea:

1. **Multi-scale critics**: Train separate $Q^k(s, a_{1:k})$ for each candidate chunk size $k \in K = \{1, 3, 5\}$, alongside $V^k(s)$ (value) and $M^k(s)$ (second moment) networks.

2. **Learned Variance Normalization (LVN)**: Instead of computing Z-scores from sample statistics (which are noisy when $N$ is small), we learn the advantage variance directly:

$$\tilde{z}^k(s, a) = \frac{Q^k(s, a) - V^k(s)}{\sqrt{M^k(s)} + \epsilon}$$

where $M^k(s) \approx \mathbb{E}_{a \sim \pi}[(A^k(s,a))^2]$ is the learned second moment.

3. **Adaptive selection**: At inference, the agent samples $N$ candidate action sequences, evaluates the normalized advantage $\tilde{z}^k$ across all chunk sizes, and selects both the best chunk size $k^*$ and the best action sequence jointly.

### Why This Matters

| Aspect | QC (Original) | AQC (Ours) |
|--------|:---:|:---:|
| Chunk size | Fixed $h$ for all states | Adaptive $k^*$ per state |
| Z-score normalization | Sample statistics (noisy when $N$ small) | Learned variance (stable) |
| Contact-rich tasks | Must use small $h$ globally | Can use large $k$ in free space, small $k$ near contacts |
| Online fine-tuning | Z-score drifts with policy change | Learned variance adapts smoothly |

### Architecture Diagram

```
                    State s_t
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Q^1(s,a_{1:1})  Q^3(s,a_{1:3})  Q^5(s,a_{1:5})
    V^1(s)          V^3(s)          V^5(s)
    M^1(s)          M^3(s)          M^5(s)
        │              │              │
        ▼              ▼              ▼
    z̃^1 = A^1/√M^1  z̃^3 = A^3/√M^3  z̃^5 = A^5/√M^5
        │              │              │
        └──────────────┼──────────────┘
                       ▼
              k* = argmax z̃^k
              (select best chunk size + action)
```

---

## Repository Structure

```
.
├── agents/
│   ├── acfql.py          # ACFQL — Q-Chunking baseline (fixed chunk)
│   ├── aqc.py            # AQC — Adaptive Q-Chunking with LVN (our method)
│   ├── acrlpd.py         # RLPD-based agent
│   └── model.py          # Shared network architectures
├── envs/                 # Environment utilities (OGBench, Robomimic)
├── utils/                # Dataset, Flax utilities, network definitions
├── evaluation.py         # Evaluation loop (supports adaptive chunk selection)
├── main.py               # Main training script (offline + online RL)
├── main_online.py        # Online-only training script
├── AQC_proposed_improvements.md   # Detailed proposal document
├── PROGRESS_REPORT.md    # Current progress and next steps
└── requirements.txt      # Dependencies
```

### Key Files

| File | Description |
|------|-------------|
| `agents/aqc.py` | **Core contribution** — AQC agent with multi-scale critics ($Q^k$, $V^k$, $M^k$) and adaptive chunk selection via learned variance normalization |
| `agents/acfql.py` | Baseline — original Q-Chunking with fixed chunk size |
| `evaluation.py` | Modified to support `sample_actions_adaptive()` — automatically detects AQC and uses adaptive chunk selection during eval |
| `main.py` | Training loop with `--chunk_sizes` flag for specifying candidate chunk sizes |

---

## How to Run

### Installation
```bash
git clone https://github.com/nonamehihi1/AQC_Finetune.git
cd AQC_Finetune
pip install -r requirements.txt
```

### Run AQC (Our Method)
```bash
MUJOCO_GL=egl python main.py \
    --agent=agents/aqc.py \
    --agent.actor_type=distill-ddpg \
    --run_group=AQC_LVN \
    --env_name=cube-triple-play-singletask-task4-v0 \
    --offline_steps=1000000 \
    --online_steps=1000000 \
    --horizon_length=5 \
    --chunk_sizes=1,3,5 \
    --seed=1
```

### Run Baseline (ACFQL — Fixed Chunk)
```bash
MUJOCO_GL=egl python main.py \
    --agent=agents/acfql.py \
    --agent.actor_type=distill-ddpg \
    --run_group=Baseline_ACFQL \
    --env_name=cube-triple-play-singletask-task4-v0 \
    --offline_steps=1000000 \
    --online_steps=1000000 \
    --horizon_length=5 \
    --seed=1
```

---

## References

- **Q-Chunking**: Li et al., "Reinforcement Learning with Action Chunking", NeurIPS 2025. [[paper](https://arxiv.org/abs/2507.07969)] [[website](https://colinqiyangli.github.io/qc/)]
- **FQL**: Park et al., "Flow Q-Learning". [[code](https://github.com/seohongpark/fql)]
- **OGBench**: Park et al., "OGBench: Benchmarking Offline Goal-Conditioned RL". [[code](https://github.com/seohongpark/ogbench)]

## Acknowledgments

This codebase is built on top of [Q-Chunking](https://github.com/ColinQiyangLi/qc) and [FQL](https://github.com/seohongpark/fql). The two `rlpd_*` folders are directly taken from [RLPD](https://github.com/ikostrikov/rlpd).
