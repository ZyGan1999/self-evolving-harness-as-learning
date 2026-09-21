# Harness Evolution as Learning

This repository contains the implementation and configurations for the Q1--Q3 experiments in *Harness Evolution as Learning*. It is a source-only release: benchmark data, model transcripts, experiment sessions, aggregated results, and generated figures are not included.

The language model is held fixed within each experiment. We vary only the local personalization harness.

## Experimental scope

| Experiment | Question | Main comparison |
|---|---|---|
| **Q1: approximation** | Which preferences are reachable through prompt-level memory, and which require a stronger control interface? | Context memory versus checker-, statistic-, and action-level harnesses |
| **Q2: generalization** | How does the number of learned memory statements, \(L\), affect preference violations? | Nine memory lengths under constant full relevance |
| **Q3: optimization** | Does a self-evolving memory reach the oracle, or plateau above it? | ACE-style updates, static references, and memory-update baselines |

The primary metric is preference violation rate on a frozen evaluation set.

## Repository layout

```text
appworld_p/          Agent, session driver, rules, feedback, and memory updaters
configs/personas/    Five personas used by the reported experiments
configs/experiments/ Frozen task splits and experiment protocols
scripts/             Experiment runners and analysis utilities
```


## Personas

| Persona | Experiment |
|---|---|
| q1_format_checksum | Q1: private note format and SMS character checksum |
| q1_signoff_running_total | Q1: SMS sign-off and per-recipient running spend total |
| q1_habitual_card | Q1: habitual-card comparison with synthetic habit observations |
| q2_memory_scale | Q2: memory-length sweep over eight preferences |
| q3_self_evolution | Q3: self-evolution over five jointly satisfiable preferences |


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

## Prepare the task pool

Experiment sessions are not distributed. Before the first run, deterministically construct the task pool from AppWorld train and development metadata:

```bash
python scripts/build_task_pool.py \
  --max-difficulty 3 --only q1_format_checksum q1_signoff_running_total q3_self_evolution

python scripts/build_task_pool.py \
  --from-cache --max-difficulty 2 --only q2_memory_scale
```

The default pool seed is 13 and the default difficulty limit is 3. The first
command loads the frozen Q1 and Q3 splits from configs/experiments/. The Q1
format/checksum split contains eight training and seven evaluation tasks.
The crossover uses six training and six evaluation tasks. Q3 uses nine training
and six evaluation tasks with no shared template families: evaluation has three
phone tasks from 0d8a4ee and three Venmo tasks from 37a8675; training uses
29caf6f, 60d0b5b, and 530b157. For these frozen splits, seed and greedy coverage
settings do not change the task lists; missing or ineligible tasks raise an error.

The second command preserves those pools and loads the frozen Q2 task
split, checking eligibility at its original difficulty limit of 2. The habitual-card runner uses its
own six fixed evaluation tasks and does not require a q1_habitual_card task pool.
The initial metadata scan may take approximately 30 minutes; the second command
reuses the cache.

## Q1: harness reachability

### Running-total configuration

The running-total checker scores successful payments only. Autofill prepares a
note before execution and commits the amount to its ledger only after a successful
API response. Failed payment attempts remain in the logs for auditing and for
preferences that explicitly score attempted actions.

The frozen crossover protocol is in `configs/experiments/q1_running_total.json`:
three seeds, six evaluation tasks, six training episodes for the learned arm,
checkpoints 0/3/6, and a 60-step budget. The SMS sign-off preference is retained
alongside running totals to preserve the original persona and feedback conditions.

The five reported Q1 preference panels come from three experimental configurations.

```bash
# Private-note format and SMS checksum
for seed in 0 1 2 3 4 5; do
  python scripts/run_exp1.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona q1_format_checksum --agent fc \
    --arms baseline oracle1 learned oracle_verifier oracle_verifier_value \
    --n-train 8 --checkpoints 0 4 8 --n-eval 7 --max-steps 50 \
    --seed "$seed" --tag _q1main
done

# SMS sign-off and per-recipient running total
for seed in 0 1 2; do
  python scripts/run_exp1.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona q1_signoff_running_total --agent fc \
    --arms baseline oracle1 learned oracle_verifier oracle_verifier_value \
           oracle_stats oracle_autofill \
    --n-train 6 --checkpoints 0 3 6 --n-eval 6 --max-steps 60 \
    --seed "$seed" --tag _q1cross
done

# Habitual-card preference, rotated across three card brands
for dominant in "American Express" Chase "Wells Fargo"; do
  python scripts/run_exp1b.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona q1_habitual_card --agent fc \
    --arms baseline oracle fulllog external gate \
    --dominant "$dominant" --schedule 0 20 60 120 \
    --attempts 3 --n-eval 6 --max-steps 30 --seed 0 --tag _m1
done
```

## Q2: memory length

The R arm requires only the target assertion pools; no donor collection is needed.
The six-rule diagnostic re-scores the same episodes, excluding SMS terseness and
payment-note presence. The reported Q2 result uses the `q2_memory_scale` R arm. Every injected statement concerns a scored preference, keeping relevance fixed at 1.0. The sweep uses nine lengths, three evaluation seeds, 12 tasks per cell, and 10 rollouts per task.

```bash
# Collect learner-induced assertion pools
for seed in 1 2; do
  python scripts/run_exp2_pool.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona q2_memory_scale --n-train 32 --seed "$seed" --tag _v1
done

# Run the R-arm length sweep
for seed in 1 2 3; do
  for length in 0 1 2 3 5 10 20 60 150; do
    python scripts/run_exp2_arms.py \
      --llm anthropic:claude-haiku-4-5-20251001 \
      --persona q2_memory_scale --arms R --l-values "$length" \
      --rollouts 10 --n-eval 12 --pool-seeds 1 2 \
      --seed "$seed" --tag _v1 --cap-distinct 55
  done
done

python scripts/aggregate_s1_seeds.py
python scripts/aggregate_s1_seeds.py --exclude sms_terse payment_has_note
```


## Q3: self-evolving memory

The main ACE trajectory uses three seeds, 24 training episodes, checkpoints at 0, 6, 12, 18, and 24 episodes, three rollouts per evaluation task, and a 12-bullet capacity.

```bash
# ACE main arm
for seed in 0 1 2; do
  python scripts/run_q3_evolve.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona q3_self_evolution --agent fc --arms selfevolve_ace \
    --n-train 24 --checkpoints 0 6 12 18 24 \
    --n-eval 6 --rollouts 3 --max-steps 50 --max-bullets 12 \
    --seed "$seed" --tag _main
done

# Static references and diagnostic arms
python scripts/run_q3_evolve.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --persona q3_self_evolution --agent fc --arms none oracle \
  --n-train 0 --checkpoints 0 --n-eval 6 --rollouts 3 \
  --max-steps 50 --seed 0 --tag _main

python scripts/run_q3_evolve.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --persona q3_self_evolution --agent fc --arms selfevolve_rewrite \
  --n-train 24 --checkpoints 0 6 12 18 24 \
  --n-eval 6 --rollouts 3 --max-steps 50 --seed 0 --tag _main

python scripts/run_q3_evolve.py \
  --llm anthropic:claude-haiku-4-5-20251001 \
  --persona q3_self_evolution --agent fc --arms external_corrective \
  --feedback-tier corrective --n-train 24 --checkpoints 0 6 12 18 24 \
  --n-eval 6 --rollouts 3 --max-steps 50 --seed 0 --tag _main

# Reflexion, TEPA, and TRACE baselines
for seed in 0 1 2; do
  python scripts/run_q3_evolve.py \
    --llm anthropic:claude-haiku-4-5-20251001 \
    --persona q3_self_evolution --agent fc --arms reflexion tepa trace \
    --n-train 24 --checkpoints 0 6 12 18 24 \
    --n-eval 6 --rollouts 3 --max-steps 50 \
    --seed "$seed" --tag _baselines
done

python scripts/compare_q3_arms.py
```


## Included entry points

| Scripts | Purpose |
|---|---|
| build_task_pool.py | Prepare task splits |
| run_exp1.py, run_exp1b.py | Run the reported Q1 comparisons |
| gate_elimination_ceiling.py | Compute the card-cycling diagnostic described in Appendix A.1 |
| run_exp2_pool.py, run_exp2_arms.py | Collect Q2 assertions and run the R-arm sweep |
| aggregate_s1_seeds.py | Aggregate Q2 counts and the six-rule re-scoring |
| run_q3_evolve.py, compare_q3_arms.py | Run and summarize Q3 methods and diagnostics |

Q1 uses three context conditions (No memory, Stated, From history) and two
control conditions: Checker rejects adds verification and retry; Harness computes
uses external statistics, computed feedback, or action-field rewriting.
Q3 TRACE retries against its own learned checks, while Q1 Checker rejects uses
the ground-truth persona checker.

TEPA retains one active precedent per key with revocation history. Its prompt
specifies the TEPA_KEYS vocabulary and permits other.* keys; recognized key
variants are mapped back to the vocabulary. TRACE uses the predicate vocabulary
and literal limits in baseline_updaters.py, with a 12-check capacity and up to
three attempts. Reflexion retains three reflections; ACE retains at most 12
bullets. Reflexion, TEPA, and TRACE filter authentication calls and redact
credential arguments before constructing updater inputs.
