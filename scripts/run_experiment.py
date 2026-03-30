"""Main experiment runner for the Verifiable Agent pipeline.

Usage:
    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --config configs/experiments/multi_model.yaml
    python scripts/run_experiment.py --config configs/default.yaml --fresh   # ignore saved memory
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env file if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from src.data.schema import ExperimentRecord
from src.evolution.evolver import Evolver
from src.evolution.failure_extractor import FailureExtractor
from src.evolution.memory_store import MemoryStore
from src.llm import create_llm
from src.proposer.proposer import Proposer
from src.responder.responder import Responder
from src.tools.registry import ToolRegistry
from src.utils.logger import get_logger, save_results_json
from src.utils.metrics import compute_metrics, format_metrics_table
from src.verifier.knowledge_base import KnowledgeBase
from src.verifier.scorer import Scorer
from src.verifier.verifier import Verifier


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_single_model(
    config: dict,
    responder_config: dict,
    proposer: Proposer,
    verifier: Verifier,
    failure_extractor: FailureExtractor,
    evolver: Evolver,
    memory: MemoryStore,
    experiment_id: str,
    logger,
) -> list[dict]:
    """Run the full pipeline for a single responder model across all epochs."""
    model_name = responder_config["model"]
    num_epochs = config["experiment"]["num_epochs"]
    n_per_type = config["experiment"]["num_probes_per_type"]
    risk_weights = config.get("risk_weights")

    responder_llm = create_llm(responder_config)
    responder = Responder(responder_llm)

    all_epoch_results = []

    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} | Model: {model_name} ===")

        # Set runtime context for trace logging
        verifier._epoch = epoch
        verifier._experiment_id = experiment_id
        verifier._model_name = model_name

        # Stage 1: Generate probes
        logger.info("Stage 1: Generating safety probes...")
        probes = proposer.generate_all(n_per_type=n_per_type, risk_weights=risk_weights)
        logger.info(f"  Generated {len(probes)} probes")

        # Stage 2: Collect responses
        logger.info("Stage 2: Querying responder model...")
        responses = responder.respond_batch(probes)
        logger.info(f"  Collected {len(responses)} responses")

        # Stage 3: Verify
        logger.info("Stage 3: Verifying responses...")
        results = verifier.verify_batch(probes, responses)

        # Compute metrics
        metrics = compute_metrics(results)
        logger.info(format_metrics_table(metrics))

        # Extract failures for evolution
        failures = failure_extractor.extract(results)
        logger.info(f"  Extracted {len(failures)} informative failures")

        # Persist raw failures to disk
        memory.append_failures(failures, epoch=epoch, experiment_id=experiment_id)

        # Self-evolution: update BOTH proposer and verifier
        if failures and epoch < num_epochs - 1:
            # Evolve Proposer: generate harder probes targeting discovered weaknesses
            logger.info("  Evolving proposer strategy...")
            new_strategies = evolver.evolve(failures, epoch=epoch)
            proposer.update_memory(
                [{"pattern": s.get("suggestion", s["pattern"]), "failure_type": s.get("failure_type", "")}
                 for s in new_strategies]
            )
            logger.info(f"  Added {len(new_strategies)} new strategies to proposer memory")

            # Evolve Verifier: inject corrective few-shot examples from failures
            logger.info("  Evolving verifier with failure corrections...")
            n_corrections = verifier.evolve(failures)
            logger.info(f"  Added {n_corrections} corrective examples to verifier")

            # Calibrate Verifier: run on synthetic data, detect biases, adjust prompt
            logger.info("  Calibrating verifier on synthetic data...")
            cal_results = verifier.calibrate(n_per_label=10, seed=epoch)
            cal_acc = cal_results.get("accuracy", 0)
            cal_biases = cal_results.get("biases", [])
            logger.info(f"  Calibration accuracy: {cal_acc:.1%}")
            if cal_biases:
                logger.info(f"  Detected biases: {cal_biases}")

            # Log calibration history for trend analysis
            memory.append_calibration_history(epoch, experiment_id, cal_results)

            # Persist evolved state to disk after each epoch
            memory.save_proposer_memory(proposer.memory)
            memory.save_matcher_few_shots(verifier.matcher.few_shot_examples)
            memory.save_decomposer_few_shots(verifier.decomposer.few_shot_examples)
            memory.save_calibration_correction(verifier.matcher.calibration_correction)

            # Flush tool cache
            if verifier.tools:
                verifier.tools.flush_cache()

            logger.info("  Memory persisted to disk")

        # Store epoch results
        epoch_data = {
            "epoch": epoch,
            "model_name": model_name,
            "metrics": metrics,
            "num_probes": len(probes),
            "num_failures": len(failures),
            "memory_stats": memory.stats(),
            "results": [r.model_dump() for r in results],
        }
        all_epoch_results.append(epoch_data)

    return all_epoch_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Verifiable Agent experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--memory-dir", default="memory", help="Directory for persistent evolution memory")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved memory, start fresh")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_config = config["experiment"]
    output_dir = args.output or exp_config.get("output_dir", "results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{exp_config['name']}_{timestamp}"
    run_dir = f"{output_dir}/{experiment_id}"

    logger = get_logger("experiment", run_dir)
    logger.info(f"Experiment: {exp_config['name']}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {run_dir}")

    # Initialize persistent memory
    memory = MemoryStore(path=args.memory_dir)

    if args.fresh:
        logger.info("--fresh flag set: starting with empty memory")
        memory.reset()
    else:
        stats = memory.stats()
        logger.info(f"Loaded persistent memory: {stats}")

    # Save config
    save_results_json(config, f"{run_dir}/config.json")

    # Initialize shared components
    proposer_llm = create_llm(config["proposer_llm"])
    verifier_llm = create_llm(config["verifier_llm"])

    # Load proposer with persisted memory
    saved_proposer_memory = memory.load_proposer_memory()
    proposer = Proposer(proposer_llm, seed=exp_config.get("seed", 42), memory=saved_proposer_memory)
    logger.info(f"Proposer initialized with {len(saved_proposer_memory)} saved strategies")

    kb_config = config.get("knowledge_base", {})
    kb = KnowledgeBase(
        path=kb_config.get("path", "knowledge_base/documents"),
        retrieval_method=kb_config.get("retrieval_method", "tfidf"),
        top_k=kb_config.get("top_k", 5),
    )
    kb.load()
    logger.info(f"Knowledge base loaded: {len(kb)} documents")

    scoring_config = config.get("scoring", {})
    scorer = Scorer(
        supported_weight=scoring_config.get("supported_weight", 1.0),
        contradicted_weight=scoring_config.get("contradicted_weight", -2.0),
        not_mentioned_weight=scoring_config.get("not_mentioned_weight", -0.5),
    )

    # Initialize external tools
    tools_config = config.get("tools", {})
    tool_registry = None
    if tools_config.get("enabled", True):
        tool_registry = ToolRegistry(
            enable_web=tools_config.get("web_search", True),
            enable_wikidata=tools_config.get("wikidata", True),
            enable_calculator=tools_config.get("calculator", True),
            enable_wikipedia=tools_config.get("wikipedia", True),
            enable_scholar=tools_config.get("semantic_scholar", True),
            enable_code_executor=tools_config.get("code_executor", True),
            memory_store=memory,
        )
        enabled = [t.name for t in tool_registry.tools]
        logger.info(f"External tools enabled: {enabled}")

    verifier = Verifier(verifier_llm, kb, scorer, tool_registry=tool_registry, memory=memory)

    # Load verifier with persisted few-shot corrections
    saved_matcher = memory.load_matcher_few_shots()
    saved_decomposer = memory.load_decomposer_few_shots()
    if saved_matcher:
        verifier.matcher.few_shot_examples = saved_matcher
        logger.info(f"Verifier matcher loaded {len(saved_matcher)} saved few-shots")
    if saved_decomposer:
        verifier.decomposer.few_shot_examples = saved_decomposer
        logger.info(f"Verifier decomposer loaded {len(saved_decomposer)} saved few-shots")
    saved_calibration = memory.load_calibration_correction()
    if saved_calibration:
        verifier.matcher.calibration_correction = saved_calibration
        logger.info("Verifier loaded saved calibration correction")

    failure_extractor = FailureExtractor()
    evolver = Evolver(proposer_llm)

    # Run for each responder model
    all_results = {}
    for responder_config in config["responder_models"]:
        model_name = responder_config["model"]
        logger.info(f"\n{'#' * 60}")
        logger.info(f"# Running model: {model_name}")
        logger.info(f"{'#' * 60}")

        model_results = run_single_model(
            config, responder_config, proposer, verifier,
            failure_extractor, evolver, memory, experiment_id, logger,
        )
        all_results[model_name] = model_results

        # Save per-model results
        save_results_json(model_results, f"{run_dir}/{model_name.replace('/', '_')}_results.json")

        # Reset evolution state between models for fair comparison,
        # but reload from persistent memory (not empty) so each model
        # starts from the same accumulated baseline
        proposer.memory = memory.load_proposer_memory()
        verifier.matcher.few_shot_examples = memory.load_matcher_few_shots()
        verifier.matcher.calibration_correction = memory.load_calibration_correction()
        verifier.decomposer.few_shot_examples = memory.load_decomposer_few_shots()

    # Save combined results
    save_results_json(all_results, f"{run_dir}/all_results.json")

    # Final memory stats
    final_stats = memory.stats()
    logger.info(f"\nFinal memory state: {final_stats}")
    logger.info(f"All results saved to {run_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
