from __future__ import annotations

from typing import Iterable, Optional

from .generator import CharacterGenerator

_DEFAULT_NEGATIVE_TERMS = {
    "duplicate",
    "text",
    "caption",
    "speech bubble",
    "watermark",
    "logo",
}
_DEFAULT_POSITIVE_HINT = "blank background, solid background, studio backdrop, uncluttered"


def _coerce_list(value: Optional[object]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        return [item for item in parts if item]
    if isinstance(value, Iterable):  # type: ignore[arg-type]
        result: list[str] = []
        for entry in value:  # type: ignore[iteration-over-optional]
            result.extend(_coerce_list(entry))
        return result
    return [str(value)]


class BulletProofGenerator:
    """Facade that applies bullet-proof prompt guards before delegating generation."""

    def __init__(self, preset: Optional[dict]) -> None:
        self.preset = preset or {}
        self._core = CharacterGenerator(self.preset)

    def _compose_prompt(self, prompt: str) -> str:
        positive_terms = _coerce_list(self.preset.get("positive"))
        if _DEFAULT_POSITIVE_HINT:
            positive_terms.append(_DEFAULT_POSITIVE_HINT)
        positive_terms = [term for term in positive_terms if term]
        if not positive_terms:
            return prompt
        extra = ", ".join(positive_terms)
        return f"{prompt}, {extra}"

    def _compose_negative(self, negative_prompt: Optional[str]) -> Optional[str]:
        negatives = set(_coerce_list(self.preset.get("negative")))
        negatives.update(_coerce_list(self.preset.get("negative_prompt")))
        negatives.update(_coerce_list(negative_prompt))
        negatives.update(_DEFAULT_NEGATIVE_TERMS)
        filtered = sorted(term for term in negatives if term)
        return ", ".join(filtered) if filtered else None

    def generate(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt")
        updated_prompt = self._compose_prompt(prompt)
        kwargs["prompt"] = updated_prompt
        kwargs["negative_prompt"] = self._compose_negative(negative_prompt)
        return self._core.generate(**kwargs)

    def refine(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt")
        kwargs["prompt"] = self._compose_prompt(prompt)
        kwargs["negative_prompt"] = self._compose_negative(negative_prompt)
        return self._core.refine(**kwargs)

    def consume_warnings(self) -> list[str]:
        return self._core.consume_warnings()

    def __getattr__(self, item):
        return getattr(self._core, item)
