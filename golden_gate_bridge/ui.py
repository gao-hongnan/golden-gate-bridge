from __future__ import annotations

import threading
from typing import Any, Type

import modal
from fastapi import FastAPI

from .config import GPU, IMAGE, VOLUME, Constants, GenerationConfig, ServingConfig, app
from .logger import get_logger
from .state import GenerationOutput, State
from .train import generate
from .utils import chat_template_unparse, load_model, load_tokenizer

with IMAGE.imports():
    import gradio as gr
    from gradio.routes import mount_gradio_app
    from repeng import ControlModel, ControlVector

logger = get_logger(__name__)


web_app = FastAPI()

IDENTIFIER: str = "20240608191148"
IDENTIFIERS: list[str] = ["20240608191148"]


class ModelSingleton:
    _model_instance: Model | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls: Type[ModelSingleton], identifier: str = IDENTIFIER) -> Model:
        if cls._model_instance is None:
            with cls._lock:
                if cls._model_instance is None:
                    cls._model_instance = Model(identifier)
        return cls._model_instance


@app.cls(
    image=IMAGE,
    gpu=GPU,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    volumes={Constants.TARGET_ARTIFACTS_DIR: VOLUME},
)
class Model:
    pretrained_model_name_or_path: str = Constants.MODEL_NAME
    device: str = "cuda:0"  # master gpu for inference

    def __init__(self, identifier: str = IDENTIFIER) -> None:
        self.identifier = identifier

    @modal.enter()
    def start_engine(self) -> None:
        """Start the engine."""
        # load state from identifier
        self.state = State.load_snapshots(
            filepath=f"{Constants.TARGET_ARTIFACTS_DIR}/{Constants.APP_NAME}/{self.identifier}/{Constants.FILE_NAME}.pt"
        )
        self.state.pretty_print()

        # load composer
        self.composer = self.state.composer
        assert self.composer is not None
        assert self.composer.registry.identifier == self.identifier
        self.composer.pretty_print()

        # load tokenizer
        self.tokenizer = load_tokenizer(self.pretrained_model_name_or_path)
        self.tokenizer.pad_token_id = self.composer.tokenizer_config.pad_token_id
        assert self.composer.generation_config.pad_token_id == self.tokenizer.eos_token_id

        # load model
        self.model = load_model(
            self.pretrained_model_name_or_path,
            device_map=self.composer.llama_config.device_map,
        )

        # load controlled vector
        self.controlled_vector = ControlVector.import_gguf(
            path=f"{self.composer.registry.save_directory}/{self.identifier}/{self.composer.registry.gguf_filename}"
        )

    @modal.method()
    def inference(self, serving_config: ServingConfig) -> GenerationOutput:
        assert self.composer is not None

        wrapped_model = self.model
        self.model = ControlModel(wrapped_model, layer_ids=self.composer.llama_config.layer_ids)
        # update composer generation config
        self.composer.generation_config = GenerationConfig(
            question=serving_config.question,
            max_new_tokens=serving_config.max_new_tokens,
            repetition_penalty=serving_config.repetition_penalty,
            temperature=serving_config.temperature,
            show_baseline=serving_config.show_baseline,
            coefficients=serving_config.coefficients,
            pad_token_id=self.tokenizer.eos_token_id,
            device=self.device,
        )
        self.composer.pretty_print()

        output = generate(
            composer=self.composer,
            model=self.model,
            tokenizer=self.tokenizer,
            input=chat_template_unparse([("user", f"{serving_config.question}")]),
            labeled_vectors=[
                (f"{coef} * bridge_vector", coef * self.controlled_vector)
                for coef in self.composer.generation_config.coefficients
            ],
        )
        return output


@app.function(
    image=IMAGE,
    timeout=int(Constants.TIMEOUT),
    container_idle_timeout=int(Constants.CONTAINER_IDLE_TIMEOUT),
    concurrency_limit=int(Constants.CONCURRENCY_LIMIT),
    keep_warm=int(Constants.KEEP_WARM),
)
@modal.asgi_app()
def ui() -> FastAPI:
    logger.info("Starting the UI...")

    def go(
        identifier: str,
        question: str,
        max_new_tokens: int,
        repetition_penalty: float,
        temperature: float,
        show_baseline: bool,
        coefficients: list[float] | str,
    ) -> dict[str, Any]:
        """Call out to the inference in a separate Modal environment with a GPU."""
        logger.info("Generating output for identifier: %s", identifier)

        if isinstance(coefficients, str):
            coefficients = [float(coef.strip()) for coef in coefficients.split(",")]
        model = Model(identifier=identifier)
        # model = ModelSingleton.get_instance(identifier=identifier)
        serving_config = ServingConfig(
            question=question,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            show_baseline=show_baseline,
            coefficients=coefficients,
        )
        output: GenerationOutput = model.inference.remote(serving_config)

        return output.model_dump(mode="json")

    logger.info("Starting Interface...")
    interface = gr.Interface(
        fn=go,
        inputs=[
            gr.Dropdown(choices=IDENTIFIERS, value=IDENTIFIER, label="Model ID"),
            gr.Textbox(
                value="What are you?",
                lines=2,
                placeholder="Enter your text here...",
                label="Question/Prompt",
            ),
            gr.Slider(minimum=32, maximum=256, value=256, label="Max New Tokens"),
            gr.Slider(
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1.25,
                label="Repetition Penalty",
            ),
            gr.Number(value=0.7, label="Temperature"),
            gr.Checkbox(label="Show Baseline"),
            gr.Textbox(
                value="0.9, 1.1",
                label="Coefficients",
                placeholder="Enter coefficients separated by commas, e.g., 0.5, 0.75, 1.0",
            ),
        ],
        outputs=gr.JSON(label="Control Model Output"),
        # outputs=gr.Textbox(label="Control Model Output"),
        # some extra bits to make it look nicer
        title="Llama-3-70B Golden Gate Bridge",
        description="# Try out the golden gate bridge with Llama-3-70B!"
        "\n\nCheck out [the code on GitHub](https://github.com/gao-hongnan/modal-examples/tree/main/golden_gate_bridge)"
        " if you want to create your own version or just see how it works."
        "\n\nPowered by [Modal](https://modal.com) 🚀",
        theme="soft",
        allow_flagging="never",
        concurrency_limit=10,
    )
    logger.info("Interface created...")
    return mount_gradio_app(app=web_app, blocks=interface, path="/")  # type: ignore[no-any-return]
