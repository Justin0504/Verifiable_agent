"""SEVA Self-Evolution Orchestrator.

Runs the full PROBE → REFLECT → REFINE → VERIFY loop for N rounds.

Each round:
  1. VERIFY: Evaluate current model on ClearFacts (+ optional benchmarks)
  2. REFLECT: Analyze failures, extract rules, build weakness profile
  3. PROBE: Generate targeted adversarial data based on weaknesses
  4. REFINE: Merge new data with existing, run GRPO training

The loop starts with VERIFY (need baseline evaluation before probing).

Usage:
    python scripts/run_self_evolution.py \
        --base-model /path/to/seva_sft_checkpoint \
        --data-dir /path/to/data \
        --output-dir /path/to/evolution \
        --rounds 3

    # Resume from a specific round
    python scripts/run_self_evolution.py \
        --base-model /path/to/round1_checkpoint \
        --output-dir /path/to/evolution \
        --start-round 2 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = Path(__file__).parent.parent


def run_cmd(cmd: list[str] | str, cwd: str = None, env: dict = None,
            timeout: int = 7200) -> tuple[int, str]:
    """Run a command and return (returncode, output)."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    if isinstance(cmd, str):
        cmd = cmd.split()

    print(f"\n  CMD: {' '.join(cmd[:10])}...")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=full_env, capture_output=True,
            text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[-500:]}")
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -1, str(e)


def phase_verify(model_path: str, round_dir: Path, benchmarks: list[str],
                 max_samples: int = None) -> dict:
    """VERIFY phase: evaluate current model on benchmarks."""
    print(f"\n{'='*60}")
    print(f"VERIFY: Evaluating model at {model_path}")
    print(f"{'='*60}")

    eval_dir = round_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "eval_seva.py"),
        "--model", model_path,
        "--benchmarks", *benchmarks,
        "--output-dir", str(eval_dir),
    ]
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])

    rc, output = run_cmd(cmd, timeout=3600)

    if rc != 0:
        print(f"  ERROR: eval failed (rc={rc})")
        return {}

    # Load summary
    summary_path = eval_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)

    return {}


def phase_reflect(round_dir: Path, benchmarks: list[str],
                  existing_bank_path: str = None,
                  epoch: int = 0, use_llm: bool = False) -> dict:
    """REFLECT phase: analyze failures, extract rules."""
    print(f"\n{'='*60}")
    print(f"REFLECT: Analyzing failures and extracting rules")
    print(f"{'='*60}")

    analysis_dir = round_dir / "analysis"
    bank_dir = round_dir / "reasoning_bank"

    # Step 1: Analyze failures for each benchmark
    weakness_profiles = {}
    for bench in benchmarks:
        results_file = round_dir / "eval" / f"{bench}_results.json"
        if not results_file.exists():
            print(f"  WARNING: {results_file} not found, skipping")
            continue

        print(f"\n  Analyzing {bench}...")
        bench_analysis = analysis_dir / bench
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "analyze_failures.py"),
            "--results-file", str(results_file),
            "--output-dir", str(bench_analysis),
        ]
        rc, _ = run_cmd(cmd)

        if rc == 0:
            profile_path = bench_analysis / "weakness_profile.json"
            if profile_path.exists():
                with open(profile_path) as f:
                    weakness_profiles[bench] = json.load(f)

    # Merge profiles across benchmarks (weighted average)
    merged_profile = merge_weakness_profiles(weakness_profiles)
    merged_path = analysis_dir / "merged_weakness_profile.json"
    with open(merged_path, "w") as f:
        json.dump(merged_profile, f, indent=2)

    # Step 2: Extract rules
    print(f"\n  Extracting verification rules...")
    # Use the primary benchmark results
    primary_bench = benchmarks[0]
    results_file = round_dir / "eval" / f"{primary_bench}_results.json"

    if results_file.exists():
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "extract_rules.py"),
            "--results-file", str(results_file),
            "--output-dir", str(bank_dir),
            "--epoch", str(epoch),
        ]
        if existing_bank_path and Path(existing_bank_path).exists():
            cmd.extend(["--existing-bank", existing_bank_path])
        if use_llm:
            cmd.append("--use-llm")

        rc, _ = run_cmd(cmd)

    return {
        "weakness_profile_path": str(merged_path),
        "bank_path": str(bank_dir / "bank.json"),
        "rules_prompt_path": str(bank_dir / "rules_prompt.txt"),
    }


def merge_weakness_profiles(profiles: dict) -> dict:
    """Merge weakness profiles from multiple benchmarks."""
    if not profiles:
        return {}

    if len(profiles) == 1:
        return list(profiles.values())[0]

    # Average targeting weights across benchmarks
    all_weights = {}
    for bench, profile in profiles.items():
        for etype, w in profile.get("targeting_weights", {}).items():
            if etype not in all_weights:
                all_weights[etype] = []
            all_weights[etype].append(w)

    merged_weights = {k: round(sum(v) / len(v), 3)
                      for k, v in all_weights.items()}

    # Average accuracy
    accs = [p.get("overall_accuracy", 0) for p in profiles.values()
            if p.get("overall_accuracy")]
    f1s = [p.get("overall_f1", 0) for p in profiles.values()
           if p.get("overall_f1")]

    # Merge error type weaknesses
    all_weaknesses = {}
    for profile in profiles.values():
        for etype, stats in profile.get("error_type_weaknesses", {}).items():
            if etype not in all_weaknesses:
                all_weaknesses[etype] = {"accuracies": [], "totals": []}
            all_weaknesses[etype]["accuracies"].append(stats.get("accuracy", 0))
            all_weaknesses[etype]["totals"].append(stats.get("total", 0))

    merged_weaknesses = {}
    for etype, data in all_weaknesses.items():
        avg_acc = sum(data["accuracies"]) / len(data["accuracies"])
        merged_weaknesses[etype] = {
            "accuracy": round(avg_acc, 3),
            "total": sum(data["totals"]),
            "priority": round(1.0 - avg_acc, 3),
        }

    return {
        "overall_accuracy": round(sum(accs) / max(len(accs), 1), 3) if accs else None,
        "overall_f1": round(sum(f1s) / max(len(f1s), 1), 3) if f1s else None,
        "error_type_weaknesses": dict(sorted(
            merged_weaknesses.items(),
            key=lambda x: x[1]["priority"], reverse=True
        )),
        "targeting_weights": merged_weights,
        "benchmarks": list(profiles.keys()),
    }


def phase_probe(round_dir: Path, weakness_profile_path: str,
                num_samples: int = 2000, model: str = "gpt-4o-mini",
                workers: int = 4) -> dict:
    """PROBE phase: generate adversarial data."""
    print(f"\n{'='*60}")
    print(f"PROBE: Generating {num_samples} adversarial samples")
    print(f"{'='*60}")

    probe_dir = round_dir / "adversarial_data"

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "generate_adversarial_probes.py"),
        "--weakness-profile", weakness_profile_path,
        "--output-dir", str(probe_dir),
        "--num-samples", str(num_samples),
        "--model", model,
        "--workers", str(workers),
        "--annotate",
    ]

    # Check for hard samples from analysis (for hard negative mining)
    hard_samples = None
    if (round_dir / "analysis").exists():
        hard_files = list((round_dir / "analysis").glob("*/hard_samples.json"))
        if hard_files:
            hard_samples = hard_files[0]

    if hard_samples and hard_samples.exists():
        cmd.extend(["--hard-samples", str(hard_samples)])

    rc, _ = run_cmd(cmd, timeout=3600)

    return {
        "probe_dir": str(probe_dir),
        "sft_data": str(probe_dir / "adversarial_sft.jsonl"),
        "grpo_data": str(probe_dir / "adversarial_grpo.parquet"),
    }


def phase_refine(round_dir: Path, model_path: str,
                 grpo_data_path: str, original_data_path: str,
                 rules_prompt_path: str = None,
                 steps: int = 200) -> dict:
    """REFINE phase: merge data and run GRPO training."""
    print(f"\n{'='*60}")
    print(f"REFINE: GRPO training ({steps} steps)")
    print(f"{'='*60}")

    checkpoint_dir = round_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Merge original + adversarial GRPO data
    merged_data = round_dir / "merged_grpo.parquet"
    merge_parquet_files(original_data_path, grpo_data_path, str(merged_data))

    # Run GRPO training
    # Note: this calls the shell script; on server, adjust paths as needed
    env = {
        "MODEL_PATH": model_path,
        "DATA_DIR": str(round_dir),
    }

    # For now, generate a round-specific training script
    train_script = round_dir / "train_grpo.sh"
    write_grpo_script(train_script, model_path, str(merged_data),
                      str(checkpoint_dir), steps, rules_prompt_path)

    print(f"  Training script: {train_script}")
    print(f"  Model: {model_path}")
    print(f"  Data: {merged_data}")
    print(f"  Steps: {steps}")

    # The actual GRPO training needs to be run on the server with GPUs
    # We just prepare everything here
    return {
        "train_script": str(train_script),
        "merged_data": str(merged_data),
        "checkpoint_dir": str(checkpoint_dir),
        "status": "prepared",  # "completed" after manual GPU execution
    }


def merge_parquet_files(original: str, adversarial: str, output: str):
    """Merge original and adversarial GRPO parquet files."""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa

        tables = []
        if Path(original).exists():
            tables.append(pq.read_table(original))
        if Path(adversarial).exists():
            tables.append(pq.read_table(adversarial))

        if tables:
            merged = pa.concat_tables(tables)
            pq.write_table(merged, output)
            print(f"  Merged {len(merged)} samples → {output}")
        else:
            print(f"  WARNING: No data to merge")

    except ImportError:
        print("  WARNING: pyarrow unavailable, copying original data")
        if Path(original).exists():
            shutil.copy2(original, output)


def write_grpo_script(path: Path, model: str, data: str,
                      output: str, steps: int,
                      rules_prompt: str = None):
    """Write a round-specific GRPO training script."""
    # Calculate epochs from steps (approximate)
    epochs = max(1, steps // 50)

    script = f"""#!/bin/bash
# SEVA v3 Self-Evolution GRPO — Auto-generated
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate verl

set -x
export CUDA_VISIBLE_DEVICES=0,1
export SEVA_REWARD_MODULE=seva_reward_v3  # Use v3 grounded reward

MODEL="{model}"
TRAIN_DATA="{data}"
EXPERIMENT_NAME="seva_evo_round_$(date +%Y%m%d_%H%M)"

kill -9 $(lsof -t -i :8000) 2>/dev/null || true

python -m verl.trainer.main_ppo \\
    --config-name='seva_grpo' \\
    data.train_files="$TRAIN_DATA" \\
    data.train_batch_size=64 \\
    data.max_prompt_length=768 \\
    data.max_response_length=512 \\
    data.truncation=left \\
    algorithm.use_kl_in_reward=False \\
    algorithm.adv_estimator=grpo \\
    actor_rollout_ref.model.path=$MODEL \\
    actor_rollout_ref.model.trust_remote_code=True \\
    actor_rollout_ref.actor.grad_clip=0.1 \\
    actor_rollout_ref.actor.optim.lr=2e-6 \\
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    trainer.logger='["console"]' \\
    trainer.project_name="seva-evo" \\
    trainer.experiment_name=$EXPERIMENT_NAME \\
    trainer.n_gpus_per_node=2 \\
    trainer.nnodes=1 \\
    trainer.save_freq=100 \\
    trainer.test_freq=20 \\
    trainer.val_before_train=True \\
    trainer.total_epochs={epochs}
"""
    with open(path, "w") as f:
        f.write(script)
    os.chmod(path, 0o755)


def check_validation_gate(current_results: dict, prev_results: dict,
                          max_regression: float = 0.05) -> tuple[bool, str]:
    """Validation gate: check that no benchmark regresses too much."""
    if not prev_results or not current_results:
        return True, "No previous results to compare"

    curr = current_results.get("results", {})
    prev = prev_results.get("results", {})

    regressions = []
    for bench in prev:
        if bench not in curr:
            continue
        prev_f1 = prev[bench].get("macro_f1", 0)
        curr_f1 = curr[bench].get("macro_f1", 0)
        delta = curr_f1 - prev_f1

        if delta < -max_regression:
            regressions.append(f"{bench}: {prev_f1:.3f} → {curr_f1:.3f} ({delta:+.3f})")

    if regressions:
        return False, f"Regressions detected: {'; '.join(regressions)}"

    return True, "All benchmarks passed validation gate"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True,
                        help="Starting model checkpoint path")
    parser.add_argument("--data-dir", type=str,
                        default=str(PROJECT_DIR / "data" / "attribution"),
                        help="Directory with original GRPO training data")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for all evolution artifacts")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of evolution rounds")
    parser.add_argument("--start-round", type=int, default=0,
                        help="Resume from this round number")
    parser.add_argument("--benchmarks", nargs="+", default=["clearfacts"],
                        help="Benchmarks for evaluation")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--probes-per-round", type=int, default=2000,
                        help="Adversarial samples per round")
    parser.add_argument("--grpo-steps", type=int, default=200,
                        help="GRPO training steps per round")
    parser.add_argument("--teacher-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use-llm-rules", action="store_true",
                        help="Use LLM for rule extraction")
    parser.add_argument("--dry-run", action="store_true",
                        help="Prepare everything but don't run GPU training")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_grpo_data = str(Path(args.data_dir) / "seva_grpo_train.parquet")
    current_model = args.base_model
    prev_eval_results = None
    bank_path = None

    # Evolution log
    evolution_log = []

    print(f"\n{'#'*60}")
    print(f"# SEVA SELF-EVOLUTION")
    print(f"# Rounds: {args.start_round} → {args.start_round + args.rounds - 1}")
    print(f"# Base model: {current_model}")
    print(f"# Benchmarks: {args.benchmarks}")
    print(f"{'#'*60}")

    for round_num in range(args.start_round, args.start_round + args.rounds):
        round_dir = output_dir / f"round{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_start = time.time()

        print(f"\n\n{'#'*60}")
        print(f"# ROUND {round_num}")
        print(f"# Model: {current_model}")
        print(f"{'#'*60}")

        # ---- VERIFY ----
        eval_results = phase_verify(
            current_model, round_dir, args.benchmarks, args.max_eval_samples
        )

        # Validation gate (skip for round 0)
        if round_num > args.start_round and prev_eval_results:
            passed, msg = check_validation_gate(eval_results, prev_eval_results)
            print(f"\n  GATE: {msg}")
            if not passed:
                print(f"  WARNING: Validation gate FAILED. Consider reverting.")
                # Don't abort — log the warning and continue

        # ---- REFLECT ----
        reflect_out = phase_reflect(
            round_dir, args.benchmarks,
            existing_bank_path=bank_path,
            epoch=round_num,
            use_llm=args.use_llm_rules,
        )
        bank_path = reflect_out.get("bank_path")
        weakness_profile_path = reflect_out.get("weakness_profile_path")

        # ---- PROBE ----
        if weakness_profile_path and Path(weakness_profile_path).exists():
            probe_out = phase_probe(
                round_dir, weakness_profile_path,
                num_samples=args.probes_per_round,
                model=args.teacher_model,
                workers=args.workers,
            )
        else:
            print("  WARNING: No weakness profile, skipping PROBE")
            probe_out = {}

        # ---- REFINE ----
        grpo_data = probe_out.get("grpo_data", "")
        if grpo_data and Path(grpo_data).exists():
            refine_out = phase_refine(
                round_dir, current_model,
                grpo_data, original_grpo_data,
                rules_prompt_path=reflect_out.get("rules_prompt_path"),
                steps=args.grpo_steps,
            )
        else:
            print("  WARNING: No adversarial data, skipping REFINE")
            refine_out = {"status": "skipped"}

        # Log round results
        round_log = {
            "round": round_num,
            "model": current_model,
            "eval_results": eval_results,
            "bank_path": bank_path,
            "probe_count": args.probes_per_round,
            "refine_status": refine_out.get("status"),
            "duration_s": round(time.time() - round_start),
        }
        evolution_log.append(round_log)

        # Save evolution log after each round
        with open(output_dir / "evolution_log.json", "w") as f:
            json.dump(evolution_log, f, indent=2, default=str)

        # Print round summary
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} SUMMARY")
        print(f"  Duration: {round_log['duration_s']}s")
        if eval_results and eval_results.get("results"):
            for bench, r in eval_results.get("results", {}).items():
                print(f"  {bench}: acc={r.get('accuracy', '?')}, "
                      f"F1={r.get('macro_f1', '?')}")
        print(f"  Refine: {refine_out.get('status')}")
        if refine_out.get("status") == "prepared":
            print(f"\n  >>> MANUAL STEP REQUIRED <<<")
            print(f"  Run on GPU server:")
            print(f"    bash {refine_out.get('train_script')}")
            print(f"  Then resume with --start-round {round_num + 1}")
            print(f"    --base-model <new_checkpoint_path>")
        print(f"{'='*60}")

        # Update current model for next round
        # (in practice, user updates this after GPU training)
        checkpoint_dir = refine_out.get("checkpoint_dir")
        if checkpoint_dir and Path(checkpoint_dir).exists():
            # Look for merged checkpoint
            final = Path(checkpoint_dir) / "final"
            if final.exists():
                current_model = str(final)

        prev_eval_results = eval_results

    # Final summary
    print(f"\n\n{'#'*60}")
    print(f"# EVOLUTION COMPLETE")
    print(f"# Rounds: {args.start_round} → {args.start_round + args.rounds - 1}")
    print(f"{'#'*60}")

    if len(evolution_log) >= 2:
        first = evolution_log[0].get("eval_results", {}).get("results", {})
        last = evolution_log[-1].get("eval_results", {}).get("results", {})
        for bench in first:
            if bench in last:
                f1_start = first[bench].get("macro_f1", 0)
                f1_end = last[bench].get("macro_f1", 0)
                print(f"  {bench}: F1 {f1_start:.3f} → {f1_end:.3f} "
                      f"({f1_end - f1_start:+.3f})")

    print(f"\nEvolution log: {output_dir / 'evolution_log.json'}")


if __name__ == "__main__":
    main()
