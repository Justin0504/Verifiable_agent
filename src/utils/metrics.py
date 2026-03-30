"""Evaluation metrics for the Verifiable Agent pipeline."""

from __future__ import annotations

from src.data.schema import ClaimLabel, ExperimentRecord, RiskType, VerificationResult


def compute_metrics(results: list[VerificationResult]) -> dict:
    """Compute comprehensive metrics from verification results.

    Returns a dict with:
    - hallucination_rate: fraction of responses with >= 1 Contradicted claim
    - avg_score: mean response-level score
    - claim_distribution: {S: n, C: n, N: n}
    - per_risk_type: breakdown by risk category
    - total_claims: total number of atomic claims
    - avg_claims_per_response: mean claims per response
    """
    if not results:
        return {"hallucination_rate": 0.0, "avg_score": 0.0}

    total_s = sum(r.num_supported for r in results)
    total_c = sum(r.num_contradicted for r in results)
    total_n = sum(r.num_not_mentioned for r in results)
    total_claims = total_s + total_c + total_n

    hallucinated_count = sum(1 for r in results if r.num_contradicted > 0)
    avg_score = sum(r.score for r in results) / len(results)

    # Per risk type breakdown
    per_risk: dict[str, dict] = {}
    for risk_type in RiskType:
        subset = [r for r in results if r.probe.risk_type == risk_type]
        if not subset:
            continue
        sub_s = sum(r.num_supported for r in subset)
        sub_c = sum(r.num_contradicted for r in subset)
        sub_n = sum(r.num_not_mentioned for r in subset)
        sub_total = sub_s + sub_c + sub_n
        per_risk[risk_type.value] = {
            "count": len(subset),
            "hallucination_rate": sum(1 for r in subset if r.num_contradicted > 0) / len(subset),
            "avg_score": sum(r.score for r in subset) / len(subset),
            "claim_distribution": {"S": sub_s, "C": sub_c, "N": sub_n},
            "s_ratio": sub_s / sub_total if sub_total > 0 else 0,
            "c_ratio": sub_c / sub_total if sub_total > 0 else 0,
            "n_ratio": sub_n / sub_total if sub_total > 0 else 0,
        }

    return {
        "hallucination_rate": hallucinated_count / len(results),
        "avg_score": avg_score,
        "total_responses": len(results),
        "total_claims": total_claims,
        "avg_claims_per_response": total_claims / len(results) if results else 0,
        "claim_distribution": {"S": total_s, "C": total_c, "N": total_n},
        "s_ratio": total_s / total_claims if total_claims > 0 else 0,
        "c_ratio": total_c / total_claims if total_claims > 0 else 0,
        "n_ratio": total_n / total_claims if total_claims > 0 else 0,
        "per_risk_type": per_risk,
    }


def format_metrics_table(metrics: dict) -> str:
    """Format metrics as a readable text table."""
    lines = [
        "=" * 60,
        "EXPERIMENT METRICS",
        "=" * 60,
        f"  Total responses:         {metrics.get('total_responses', 0)}",
        f"  Total claims:            {metrics.get('total_claims', 0)}",
        f"  Avg claims/response:     {metrics.get('avg_claims_per_response', 0):.1f}",
        f"  Hallucination rate:      {metrics.get('hallucination_rate', 0):.1%}",
        f"  Average score:           {metrics.get('avg_score', 0):.3f}",
        "",
        "  Claim distribution:",
        f"    Supported (S):         {metrics.get('s_ratio', 0):.1%}",
        f"    Contradicted (C):      {metrics.get('c_ratio', 0):.1%}",
        f"    Not Mentioned (N):     {metrics.get('n_ratio', 0):.1%}",
    ]

    per_risk = metrics.get("per_risk_type", {})
    if per_risk:
        lines.append("")
        lines.append("  Per risk type:")
        for risk, data in per_risk.items():
            lines.append(f"    {risk}:")
            lines.append(f"      Count:               {data['count']}")
            lines.append(f"      Hallucination rate:   {data['hallucination_rate']:.1%}")
            lines.append(f"      Avg score:            {data['avg_score']:.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)
