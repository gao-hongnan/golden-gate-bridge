from __future__ import annotations

import json
import os
from typing import Any

from .config import IMAGE, Constants

with IMAGE.imports():
    import torch
    from repeng import DatasetEntry
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        PreTrainedTokenizerFast,
    )


def load_tokenizer(
    model_name: str, **kwargs: Any
) -> PreTrainedTokenizerBase | PreTrainedTokenizerFast:
    """Load tokenizer from huggingface.

    Parameters
    ----------
    model_name : str
        Model name to load from huggingface.
    kwargs : Any
        Additional keyword arguments to pass to the tokenizer.

    Returns
    -------
    PreTrainedTokenizerBase | PreTrainedTokenizerFast
        Tokenizer loaded from huggingface.
    """

    cache_dir = kwargs.pop("cache_dir", Constants.CACHE_DIR)
    return AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, **kwargs
    )


def load_model(model_name: str, **kwargs: Any) -> PreTrainedModel:
    """Load model from huggingface.

    Parameters
    ----------
    model_name : str
        Model name to load from huggingface.
    kwargs : Any
        Additional keyword arguments to pass to the model.

    Returns
    -------
    PreTrainedModel
        Decoder for Causal Modeling loaded from huggingface.
    """
    torch_dtype = kwargs.pop("torch_dtype", torch.float16)
    cache_dir = kwargs.pop("cache_dir", Constants.CACHE_DIR)

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        **kwargs,
    )


def load_suffixes(suffix_filepath: str | os.PathLike[str]) -> list[str]:
    """Load suffixes from file.

    Parameters
    ----------
    suffix_filepath : str | os.PathLike
        File path to load suffixes from.

    Returns
    -------
    list[str]
        List of suffixes to pass into the dataset template.
    """
    with open(suffix_filepath, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def make_dataset(
    template: str,
    positive_personas: list[str],
    negative_personas: list[str],
    suffixes: list[str],
) -> list[DatasetEntry]:
    """Make dataset from template and personas.

    Parameters
    ----------
    template : str
        Template to use for dataset.
    positive_personas : list[str]
        List of positive personas.
    negative_personas : list[str]
        List of negative personas.
    suffixes : list[str]
        List of suffixes to append to the template.

    Returns
    -------
    list[DatasetEntry]
        List of dataset entries of the form (positive, negative) as a dataclass
        from `repeng`.
    """
    dataset = []
    for suffix in suffixes:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{positive_template}{suffix}",
                    negative=f"{negative_template}{suffix}",
                )
            )
    return dataset


def chat_template_unparse(messages: list[tuple[str, str]]) -> str:
    r"""Unparse chat messages into a single string.

    Parameters
    ----------
    messages : list[tuple[str, str]]
        List of tuples of the form (role, content) where role is the user
        and content is the message.

    Returns
    -------
    str
        Unparsed chat messages as a single string.

    Example
    -------
    >>> chat_template_unparse([("user", "What are you?")])
    '<|start_header_id|>user<|end_header_id|>\n\nWhat are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    """
    template = []

    for role, content in messages:
        template.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    if messages[-1][0] != "assistant":
        # prefill assistant prefix
        template.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(template)


def print_layers(model: PreTrainedModel) -> None:
    """Print named modules of the model.

    Parameters
    ----------
    model : PreTrainedModel
        Model to print named modules of.
    """

    if hasattr(model, "model"):  # mistral-like
        _layers = model.model.layers
    elif hasattr(model, "transformer"):  # gpt-2-like
        _layers = model.transformer.h
    else:
        raise ValueError(f"don't know how to get layer list for {type(model)}")

    for i, layer in enumerate(_layers):
        print(f"Layer {i}: {layer}")
