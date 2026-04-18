from __future__ import annotations

from pydantic import BaseModel, Field


class TextPayload(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    max_length: int = Field(default=256, ge=16, le=512)


class LabelProb(BaseModel):
    label: int
    label_text: str
    probability: float


class ClassificationResponse(BaseModel):
    top1: LabelProb
    top5: list[LabelProb]
    input_length_tokens: int
    truncated: bool
    model_version: str
    model_name: str
    request_id: str


class LabelEntry(BaseModel):
    label: int
    label_text: str
