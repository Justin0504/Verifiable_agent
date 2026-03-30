"""Responder: queries the model under evaluation and collects raw responses."""

from __future__ import annotations

import uuid

from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.schema import Probe, Response
from src.llm.base import BaseLLM


class Responder:
    """Stage 2 — sends probes to the target model and records responses."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.model_name = llm.model

    def respond(self, probe: Probe) -> Response:
        """Get a single response from the target model."""
        result = self.llm.generate(probe.question)
        return Response(
            probe_id=probe.id,
            model_name=self.model_name,
            text=result.text,
            tokens_used=result.input_tokens + result.output_tokens,
            latency_ms=result.latency_ms,
        )

    def respond_batch(self, probes: list[Probe], show_progress: bool = True) -> list[Response]:
        """Get responses for a batch of probes."""
        responses: list[Response] = []
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"Querying {self.model_name}...", total=len(probes)
                )
                for probe in probes:
                    resp = self.respond(probe)
                    responses.append(resp)
                    progress.advance(task)
        else:
            for probe in probes:
                responses.append(self.respond(probe))
        return responses
