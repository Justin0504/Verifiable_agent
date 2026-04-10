"""SEVA v2 output format: multi-granularity diagnosis + traceable reasoning chains.

Defines the structured output schema that SEVA produces for each (claim, source) pair.
This is the core innovation: instead of binary label, SEVA outputs:
  1. Evidence alignment: which spans in the claim map to which spans in the source
  2. Reasoning chain: step-by-step verification grounded to source sentences
  3. Error diagnosis: type of error and fix suggestion (if Not Attributable)
"""

# Error taxonomy — 6 fine-grained categories
ERROR_TYPES = {
    "numerical_exaggeration": "A number, percentage, or quantity is inflated or deflated",
    "negation_flip": "A negation is added or removed, reversing the meaning",
    "scope_inflation": "A specific claim is generalized beyond what the source supports",
    "temporal_shift": "A temporal qualifier is dropped or altered",
    "entity_substitution": "A named entity is swapped for a different one",
    "fabrication": "The claim introduces information entirely absent from the source",
}

# System prompt for SEVA v2 — instructs model to produce structured output
SEVA_SYSTEM_PROMPT = """\
You are SEVA, a fact attribution verifier. Given a claim and a source document, \
determine whether the claim is attributable to (supported by) the source.

You MUST respond with a JSON object containing:

1. "evidence_alignment": For each key phrase in the claim, find the corresponding \
evidence in the source. Each entry has:
   - "claim_span": the phrase from the claim
   - "source_span": the matching phrase from the source (or "NOT_FOUND")
   - "status": "match" | "mismatch" | "not_found"

2. "reasoning_chain": Step-by-step verification. Each step has:
   - "step": step number
   - "claim_part": what part of the claim you are checking
   - "source_evidence": the specific source sentence or phrase used
   - "judgment": "supported" | "not_supported" | "partially_supported"
   - "explanation": brief explanation of your judgment

3. "label": "Attributable" or "Not Attributable"

4. "confidence": 0.0 to 1.0

5. "error_type": (only if Not Attributable) one of: numerical_exaggeration, \
negation_flip, scope_inflation, temporal_shift, entity_substitution, fabrication

6. "fix_suggestion": (only if Not Attributable) how to modify the claim to make \
it attributable

Respond with JSON only. Be precise in span extraction."""


SEVA_USER_TEMPLATE = """\
Claim: {claim}

Source: {source}

Verify this claim against the source. Respond with the full structured JSON."""


# Simplified system prompt for SFT data generation (teacher)
TEACHER_SYSTEM_PROMPT = """\
You are an expert fact-checker creating training data for an attribution verification \
model. Given a claim, source, and gold label, generate a detailed structured analysis.

Your analysis must be:
1. ACCURATE: evidence_alignment spans must actually appear in the claim and source
2. PRECISE: reasoning steps must be logically sound and grounded
3. CONSISTENT: the reasoning chain must lead to the given gold label
4. REALISTIC: error_type and fix_suggestion must be specific and actionable

Output a single JSON object (no markdown, no explanation outside the JSON)."""


TEACHER_USER_TEMPLATE = """\
Claim: {claim}

Source: {source}

Gold label: {label}

Generate the structured analysis JSON with: evidence_alignment, reasoning_chain, \
label, confidence, error_type (if Not Attributable), fix_suggestion (if Not Attributable).

Important: the "label" field MUST be "{label}". Make the reasoning chain support this label."""
