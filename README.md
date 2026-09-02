# Harness Evolution as Learning

This repository contains the implementation and configurations for the Q1--Q3 experiments in *Harness Evolution as Learning*. It is a source-only release: benchmark data, model transcripts, experiment sessions, aggregated results, and generated figures are not included.

The language model is held fixed within each experiment. We vary only the local personalization harness.

## Experimental scope

| Experiment | Question | Main comparison |
|---|---|---|
| **Q1: approximation** | Which preferences are reachable through prompt-level memory, and which require a stronger control interface? | Context memory versus checker-, statistic-, and action-level harnesses |
| **Q2: generalization** | How does the number of learned memory statements, \(L\), affect preference violations? | Nine memory lengths under constant full relevance |
| **Q3: optimization** | Does a self-evolving memory reach the oracle, or plateau above it? | ACE-style updates, static references, and memory-update baselines |

The primary metric is preference violation rate on a frozen evaluation set. Task goal completion (TGC) is recorded as a guardrail.

## Repository layout

```text
appworld_p/          Agent, session driver, rules, feedback, and memory updaters
configs/personas/    Six personas used by the reported experiments
scripts/             Experiment runners and analysis utilities
tests/               Unit tests for the released execution paths
```


## Installation

Python 3.11 is recommended.

```bash
conda create -n harness-repro python=3.11 -y
conda activate harness-repro
pip install -r requirements.txt
appworld install
appworld download data --root appworld_root
```

Create the local environment file and provide the credentials for the selected endpoint:

```bash
cp .env.example .env
```

The reported configuration uses the model identifier
`anthropic:claude-haiku-4-5-20251001`. `ANTHROPIC_BASE_URL` may point to an Anthropic-compatible endpoint.

Verify connectivity:

```bash
python scripts/test_llm.py anthropic:claude-haiku-4-5-20251001
```

## Prepare the task pool

Experiment sessions are not distributed. Before the first run, deterministically construct the task pool from AppWorld train and development metadata:

```bash
python scripts/build_task_pool.py \
  --only p3_fc p4_q1 p5_q1b p7_offtopic p8_q2s1 p13_mixed
```

The default pool seed is 13. Building the pool scans the AppWorld task metadata and may take approximately 30 minutes.

After data and task-pool preparation, run the test suite:

```bash
python -m pytest tests/ -q
```

## Q1: harness reachability

The five reported Q1 preference panels come from three experimental configurations.

```bash
# Private-note format and SMS checksum
for seed in 0 1 2 3 4 5; do
  python scripts/run_exp1.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona p4_q1 --agent fc \
    --arms baseline oracle1 learned oracle_verifier oracle_verifier_value \
    --n-train 8 --checkpoints 0 4 8 --max-steps 50 \
    --seed "$seed" --tag _q1main
done

# SMS sign-off and per-recipient running total
for seed in 0 1 2; do
  python scripts/run_exp1.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona p5_q1b --agent fc \
    --arms baseline oracle1 learned oracle_verifier oracle_verifier_value \
           oracle_stats oracle_autofill \
    --n-train 8 --checkpoints 0 4 8 --max-steps 50 \
    --seed "$seed" --tag _q1cross
done

# Habitual-card preference, rotated across three card brands
for dominant in "American Express" Chase "Wells Fargo"; do
  python scripts/run_exp1b.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona p3_fc --agent fc \
    --arms baseline oracle fulllog external gate \
    --dominant "$dominant" --schedule 0 20 60 120 \
    --attempts 3 --n-eval 6 --max-steps 30 --seed 0 --tag _m1
done
```

## Q2: memory length

The reported Q2 result uses the `p8_q2s1` R arm. Every injected statement concerns a scored preference, keeping relevance fixed at 1.0. The sweep uses nine lengths, three evaluation seeds, 12 tasks per cell, and 10 rollouts per task.

```bash
# Collect learner-induced assertion pools
for seed in 1 2; do
  python scripts/run_exp2_pool.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona p8_q2s1 --n-train 32 --seed "$seed" --tag _v1
done

python scripts/run_exp2_pool.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --persona p7_offtopic --n-train 32 --seed 1 --tag _d1

# Run the R-arm length sweep
for seed in 1 2 3; do
  for length in 0 1 2 3 5 10 20 60 150; do
    python scripts/run_exp2_arms.py \
      --llm anthropic:claude-haiku-4-5-20251001 \
      --persona p8_q2s1 --arms R --l-values "$length" \
      --rollouts 10 --n-eval 12 --pool-seeds 1 2 \
      --seed "$seed" --tag _v1 --cap-distinct 55
  done
done

python scripts/aggregate_s1_seeds.py
```


## Q3: self-evolving memory

The main ACE trajectory uses three seeds, 24 training episodes, checkpoints at 0, 6, 12, 18, and 24 episodes, three rollouts per evaluation task, and a 12-bullet capacity.

```bash
# ACE main arm
for seed in 0 1 2; do
  python scripts/run_q3_evolve.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona p13_mixed --agent fc --arms selfevolve_ace \
    --n-train 24 --checkpoints 0 6 12 18 24 \
    --n-eval 6 --rollouts 3 --max-steps 50 --max-bullets 12 \
    --seed "$seed" --tag _p13
done

# Static references and diagnostic arms
python scripts/run_q3_evolve.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --persona p13_mixed --agent fc --arms none oracle \
  --n-train 0 --checkpoints 0 --n-eval 6 --rollouts 3 \
  --max-steps 50 --seed 0 --tag _p13

python scripts/run_q3_evolve.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --persona p13_mixed --agent fc --arms selfevolve_rewrite \
  --n-train 24 --checkpoints 0 6 12 18 24 \
  --n-eval 6 --rollouts 3 --max-steps 50 --seed 0 --tag _p13

python scripts/run_q3_evolve.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --persona p13_mixed --agent fc --arms external_corrective \
  --feedback-tier corrective --n-train 24 --checkpoints 0 6 12 18 24 \
  --n-eval 6 --rollouts 3 --max-steps 50 --seed 0 --tag _p13

# Reflexion, TEPA, and TRACE baselines
for seed in 0 1 2; do
  python scripts/run_q3_evolve.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona p13_mixed --agent fc --arms reflexion tepa trace \
    --n-train 24 --checkpoints 0 6 12 18 24 \
    --n-eval 6 --rollouts 3 --max-steps 50 \
    --seed "$seed" --tag _p13bl
done

# Backfill n=1,2,3,4 on the saved ACE memory trajectories
python scripts/backfill_q3_checkpoints.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --seeds 0 1 2 --at 1 2 3 4 --rollouts 3

python scripts/compare_q3_arms.py
```
