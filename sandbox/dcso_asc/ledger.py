"""Experiment ledger helper -- appends JSONL records to experiment_ledger.jsonl."""
import json
import time
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent / "experiment_ledger.jsonl"


def log_experiment(exp_id, mechanism, params, seeds, instance_count, scenario_scores,
                    aggregate_score, min_score, runtime_s, errors, decision, source_hash=None, notes=""):
    record = dict(
        experiment_id=exp_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_hash=source_hash,
        mechanism=mechanism,
        parameters=params,
        seeds=seeds,
        instance_count=instance_count,
        scenario_scores=scenario_scores,
        aggregate_score=aggregate_score,
        min_score=min_score,
        runtime_s=runtime_s,
        errors=errors,
        decision=decision,
        notes=notes,
    )
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
