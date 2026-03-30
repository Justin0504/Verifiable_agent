"""SciFact benchmark loader.

SciFact (Wadden et al., 2020) is a dataset of 1.4K scientific claims
paired with evidence abstracts from research papers. Each claim is
labeled as SUPPORTS, REFUTES, or NOT ENOUGH INFO.

HuggingFace: allenai/scifact
"""

from __future__ import annotations

from .base import BenchmarkLoader, BenchmarkSample


class SciFactLoader(BenchmarkLoader):
    """Load SciFact benchmark for scientific claim verification.

    Each sample has:
    - A scientific claim
    - Evidence abstracts from papers
    - Label: SUPPORTS / REFUTES / NOT_ENOUGH_INFO
    """

    name = "scifact"
    description = "1.4K scientific claims with paper evidence and S/R/NEI labels"

    def load(self, split: str = "validation", limit: int | None = None) -> list[BenchmarkSample]:
        datasets = self._try_import_datasets()

        ds = datasets.load_dataset("allenai/scifact", split=split)

        samples = []
        for i, row in enumerate(ds):
            if limit and i >= limit:
                break

            claim = row.get("claim", "")
            evidence_dict = row.get("evidence", {})
            cited_docs = row.get("cited_doc_ids", [])

            # Extract evidence sentences
            evidence_texts = []
            label = "N"  # default

            if isinstance(evidence_dict, dict):
                for doc_id, sentences_data in evidence_dict.items():
                    if isinstance(sentences_data, list):
                        for sent_data in sentences_data:
                            if isinstance(sent_data, dict):
                                lbl = sent_data.get("label", "")
                                sents = sent_data.get("sentences", [])
                                if lbl == "SUPPORT":
                                    label = "S"
                                elif lbl == "CONTRADICT":
                                    label = "C"
                                for s in sents:
                                    evidence_texts.append(str(s))
            elif isinstance(evidence_dict, list):
                for ev in evidence_dict:
                    if isinstance(ev, str):
                        evidence_texts.append(ev)

            # Map labels
            raw_label = row.get("label", "")
            if isinstance(raw_label, str):
                if raw_label.upper() in ("SUPPORT", "SUPPORTS"):
                    label = "S"
                elif raw_label.upper() in ("CONTRADICT", "REFUTES", "REFUTE"):
                    label = "C"
                elif raw_label.upper() in ("NOT_ENOUGH_INFO", "NEI"):
                    label = "N"

            sample = BenchmarkSample(
                id=f"sf_{i:04d}",
                question=claim,
                reference_answer="",
                gold_label=label,
                evidence=evidence_texts,
                metadata={
                    "risk_type": "missing_evidence",
                    "cited_doc_ids": cited_docs if isinstance(cited_docs, list) else [],
                },
            )
            samples.append(sample)

        return samples

    def load_manual_sample(self, limit: int = 20) -> list[BenchmarkSample]:
        """Fallback: curated SciFact-style scientific claims."""
        items = [
            {
                "claim": "CRISPR-Cas9 can be used to edit human embryo DNA.",
                "label": "S",
                "evidence": ["Studies have demonstrated CRISPR-Cas9 editing of human embryos, though with ethical concerns and off-target effects."],
            },
            {
                "claim": "Vitamin C supplementation prevents the common cold.",
                "label": "C",
                "evidence": ["Meta-analyses show that regular vitamin C supplementation does not reduce the incidence of colds in the general population, though it may slightly reduce duration."],
            },
            {
                "claim": "mRNA vaccines alter human DNA.",
                "label": "C",
                "evidence": ["mRNA vaccines do not enter the cell nucleus where DNA is stored and cannot alter human DNA. The mRNA is degraded after the protein is produced."],
            },
            {
                "claim": "Exercise reduces the risk of developing Alzheimer's disease.",
                "label": "S",
                "evidence": ["Multiple longitudinal studies show that regular physical exercise is associated with a 30-40% reduced risk of developing Alzheimer's disease."],
            },
            {
                "claim": "Antibiotics are effective against viral infections.",
                "label": "C",
                "evidence": ["Antibiotics target bacterial infections and are ineffective against viruses. Antiviral medications are used for viral infections."],
            },
            {
                "claim": "The gut microbiome influences mental health.",
                "label": "S",
                "evidence": ["Research has established a gut-brain axis, showing that gut microbiota composition affects neurotransmitter production and is linked to anxiety and depression."],
            },
            {
                "claim": "Stem cells can differentiate into any cell type in the human body.",
                "label": "S",
                "evidence": ["Embryonic stem cells are pluripotent and can differentiate into virtually any cell type. Adult stem cells are more limited in their differentiation potential."],
            },
            {
                "claim": "Homeopathic medicines have effects beyond placebo.",
                "label": "C",
                "evidence": ["Systematic reviews and meta-analyses, including those by the Cochrane Collaboration, have found no reliable evidence that homeopathy is effective beyond placebo."],
            },
            {
                "claim": "Climate change increases the frequency of extreme weather events.",
                "label": "S",
                "evidence": ["IPCC reports conclude with high confidence that human-induced climate change has increased the frequency and intensity of heat extremes, heavy precipitation events, and droughts."],
            },
            {
                "claim": "Glyphosate causes cancer in humans.",
                "label": "N",
                "evidence": ["The evidence is mixed: IARC classified glyphosate as 'probably carcinogenic' but EPA and EFSA concluded it is not likely carcinogenic at typical exposure levels."],
            },
            {
                "claim": "Sleep deprivation impairs immune function.",
                "label": "S",
                "evidence": ["Studies show that sleep deprivation reduces natural killer cell activity and cytokine production, increasing susceptibility to infections."],
            },
            {
                "claim": "Microplastics have been found in human blood.",
                "label": "S",
                "evidence": ["A 2022 study published in Environment International detected microplastics in 80% of human blood samples tested."],
            },
            {
                "claim": "Acupuncture activates specific neural pathways for pain relief.",
                "label": "S",
                "evidence": ["Neuroimaging studies show that acupuncture stimulation activates brain regions involved in pain modulation, including the periaqueductal gray and anterior cingulate cortex."],
            },
            {
                "claim": "5G radio waves cause COVID-19.",
                "label": "C",
                "evidence": ["COVID-19 is caused by the SARS-CoV-2 virus. Radio waves, including 5G frequencies, cannot create or spread viruses."],
            },
            {
                "claim": "Intermittent fasting improves insulin sensitivity.",
                "label": "S",
                "evidence": ["Clinical trials have shown that intermittent fasting regimens can improve insulin sensitivity and reduce fasting insulin levels in both healthy and obese individuals."],
            },
        ]

        samples = []
        for i, item in enumerate(items[:limit]):
            samples.append(BenchmarkSample(
                id=f"sf_{i:04d}",
                question=item["claim"],
                reference_answer="",
                gold_label=item["label"],
                evidence=item["evidence"],
                metadata={"risk_type": "missing_evidence"},
            ))
        return samples
