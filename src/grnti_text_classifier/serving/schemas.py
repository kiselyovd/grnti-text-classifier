"""Pydantic request/response schemas for the /classify endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextPayload(BaseModel):
    """Request body for text classification — raw abstract + optional token budget."""

    text: str = Field(min_length=1, max_length=20_000)
    max_length: int = Field(default=256, ge=16, le=512)


class LabelProb(BaseModel):
    """GRNTI class identifier together with its human-readable name and probability."""

    label: int
    label_text: str
    probability: float


class ClassificationResponse(BaseModel):
    """Response payload of `/classify`: top-1 plus top-5 probabilities and metadata."""

    top1: LabelProb
    top5: list[LabelProb]
    input_length_tokens: int
    truncated: bool
    model_version: str
    model_name: str
    request_id: str


class LabelEntry(BaseModel):
    """Label catalog entry returned by `/labels` — numeric id plus human-readable name."""

    label: int
    label_text: str
