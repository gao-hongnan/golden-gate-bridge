from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel, Field

from .config import IMAGE, Composer, GenerationConfig

with IMAGE.imports():
    import torch


class State(BaseModel):
    """Base class for state."""

    answers: list[dict[str, str]] = Field(default_factory=list)
    controlled_vector: Any = Field(default=None)  # ControlVector

    composer: Composer | None = Field(default=None)

    def save_snapshots(self, filepath: str) -> None:
        """Save the state dictionaries of the components to a file."""
        state = {
            "controlled_vector": (
                self.controlled_vector if self.controlled_vector else None
            ),
            "answers": self.answers,
            "composer": self.composer.model_dump() if self.composer else None,
        }
        torch.save(state, filepath)

    @classmethod
    def load_snapshots(cls: Type[State], filepath: str) -> State:
        """Load the state dictionaries of the components from a file."""
        state = torch.load(filepath)

        controlled_vector = (
            state["controlled_vector"] if state["controlled_vector"] else None
        )
        answers = state["answers"]
        composer = (
            Composer.model_validate(state["composer"])
            if state["composer"]
            else None
        )
        return cls(
            controlled_vector=controlled_vector,
            answers=answers,
            composer=composer,
        )

    def pretty_print(self) -> None:
        """Pretty print the configuration."""
        from rich.pretty import pprint

        pprint(self)

    class Config:
        """`torch.nn.Module` is not a supported serializable type by pydantic
        so we add `arbitrary_types_allowed = True` to allow it."""

        arbitrary_types_allowed = True
        use_enum_values = True


class GenerationOutput(BaseModel):
    """Base class for model output. No good way to pass selected fields to fastapi
    endpoint for now, so duplicate some fields here.

    See https://stackoverflow.com/questions/78527525/allow-only-certain-fields-of-pydantic-model-to-be-passed-to-fastapi-endpoint
    """

    model_name: str
    identifier: str
    modal_version: str
    app_name: str
    answers: list[dict[str, str]] = Field(default_factory=list)
    generation_config: GenerationConfig

    class Config:
        """Pydantic configuration."""

        protected_namespaces = ()
