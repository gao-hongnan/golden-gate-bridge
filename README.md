# Golden Gate Bridge

The configuration, state, constants, and initialization settings for the project
are currently housed in the `golden_gate_bridge/config.py` file. Presently, any
modifications to the configuration require manual updates directly within the
Python file. To streamline this process and add flexibility, integrating
[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) can add command
line override alongside with a yaml config - note we still can keep the config
(inherited from Pydantic's `BaseModel`) for type checking and data validation
even if we use yaml-based configuration.

To run the project, we first set up the environment by installing the required
dependencies required by Modal.

```bash
pip install modal
python3 -m modal setup
```

## Train

To train the control vector, we execute the following command:

```bash
export ALLOW_WANDB=true # optional if you want to use Weights and Biases
modal run --detach golden_gate_bridge.train --suffix-filepath=./golden_gate_bridge/data/all_truncated_outputs.json
```

## Serve

To serve the model, we execute the following command:

```bash
modal serve golden_gate_bridge.serve
```

## Deploy API

To deploy/re-deploy the model, we execute the following command:

```bash
modal deploy golden_gate_bridge.serve
```

And we can query the api by running the following sample command:

```bash
curl -X 'POST' \
  'https://gao-hongnan--golden-gate-bridge-repeng-web.modal.run/api/v1/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What are you?",
  "max_new_tokens": 256,
  "repetition_penalty": 1.25,
  "temperature": 0.7,
  "show_baseline": false,
  "coefficients": [
    0.9,
    1.1
  ]
}'
```

One can access the swagger ui [here](https://gao-hongnan--golden-gate-bridge-repeng-web.modal.run/docs).

## Serve/Deploy Gradio

To serve the Gradio interface, we execute the following command:

```bash
modal serve golden_gate_bridge.ui
```

To deploy the Gradio interface, we execute the following command:

```bash
modal deploy golden_gate_bridge.ui
```

One can access the live link [here](https://gao-hongnan--golden-gate-bridge-repeng-ui.modal.run/).

## CI Checks

```bash
ruff check
ruff format --check
dmypy run -- golden_gate_bridge
```

For typing, I did not use `internals/` and added my own `mypy.ini` file, the configurations
is as follows:

<details>
<summary>mypy config</summary>

````ini
# Reference:
# https://github.com/openai/openai-python/blob/main/mypy.ini
# https://github.com/pytorch/pytorch/blob/main/mypy.ini
[mypy]
pretty=True
show_error_codes=True
python_version=3.11

strict_equality=True
implicit_reexport=True
check_untyped_defs=True
no_implicit_optional=True

warn_return_any=True
warn_unreachable=True
warn_unused_configs=True

# Turn these options off as it could cause conflicts with the Pyright options.
warn_unused_ignores=False
warn_redundant_casts=False

disallow_any_generics=True
disallow_untyped_defs=True
disallow_untyped_calls=False
disallow_subclassing_any=True
disallow_incomplete_defs=True
disallow_untyped_decorators=True
cache_fine_grained=True

# By default, mypy reports an error if you assign a value to the result
# of a function call that doesn't return anything. We do this in our test
# cases:
# ```
# result = ...
# assert result is None
# ```
# Changing this codegen to make mypy happy would increase complexity
# and would not be worth it.
disable_error_code=func-returns-value

# https://github.com/python/mypy/issues/12162
[mypy.overrides]
module="black.files.*"
ignore_errors=true
ignore_missing_imports=true

# Third party dependencies that don't have types.
[mypy-matplotlib.*]
ignore_missing_imports=True

[mypy-mpl_toolkits.*]
ignore_missing_imports=True

[mypy-seaborn.*]
ignore_missing_imports=True

[mypy-sklearn.*]
ignore_missing_imports=True

[mypy-repeng.*]
ignore_missing_imports=True

[mypy-transformers.*]
ignore_missing_imports=True

[mypy-datasets.*]
ignore_missing_imports=True

[mypy-huggingface_hub.*]
ignore_missing_imports=True
````

</details>

## Volume

To view your artifacts, you can use the following command:

```bash
modal volume ls artifacts-volume /golden-gate-bridge-repeng
```

This will show you the contents of the volume:

| filename                                                          | type | created/modified     | size      |
| ----------------------------------------------------------------- | ---- | -------------------- | --------- |
| golden-gate-bridge-repeng/20240605160831                          | dir  | 2024-06-05 16:12 +08 | 80 B      |
| golden-gate-bridge-repeng/controlled_golden_gate_bridge_repeng.pt | file | 2024-06-04 19:50 +08 | 131.4 GiB |

And we can further check the contents of the directory `golden-gate-bridge-repeng/20240605160831` if we run the following command:

```bash
modal volume ls artifacts-volume golden-gate-bridge-repeng/20240605160831
```

| filename                                                                           | type | created/modified     | size    |
| ---------------------------------------------------------------------------------- | ---- | -------------------- | ------- |
| golden-gate-bridge-repeng/20240605160831/controlled_golden_gate_bridge_repeng.pt   | file | 2024-06-05 16:12 +08 | 3.7 MiB |
| golden-gate-bridge-repeng/20240605160831/controlled_golden_gate_bridge_repeng.gguf | file | 2024-06-05 16:12 +08 | 2.5 MiB |

## CHANGELOG

Latest app-id@ap-w6lsbMKBgeKFkNyIRSzg6x with git commit hash@664e2002a04d5a102c81f0d5d0f58a01e0468b93 trained and ran on a grid
from $0.6$ to $1.1$ with step size $0.05$ for the coefficients. This
serves as a baseline for the model. The identifier for the run is `20240608191148`.

The app logs are not available to everyone, so here I include a link to the
public weights and biases [logs here](https://wandb.ai/hongnangao/golden-gate-bridge-repeng/runs/alqffe07/logs?nw=nwuserhongnangao).

## References And Further Readings

First, a special thanks to the Modal team for their prompt support and guidance
in the slack channel. A shoutout to Charles Frye for his assistance in helping
me get started with Modal. His enthusiasm is through the roof lol.

- [Serverless Deployment of Mistral 7B with Modal Labs and HuggingFace](https://blog.premai.io/serverless-deployment-using-huggingface-and-modal/)
- [Pet Art Dreambooth with Hugging Face and Gradio](https://modal.com/docs/examples/dreambooth_app)
- [Repeng](https://github.com/vgel/repeng/blob/main/repeng/)
