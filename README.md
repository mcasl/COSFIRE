# COSFIRE

Python implementation of **COSFIRE** (*Combination of Shifted Filter Responses*)
trainable filters for keypoint detection and pattern recognition, by
G. Azzopardi and N. Petkov:

> G. Azzopardi and N. Petkov, "Trainable COSFIRE filters for keypoint detection
> and pattern recognition", *IEEE Transactions on Pattern Analysis and Machine
> Intelligence*, vol. 35(2), pp. 490-503, 2013.

A COSFIRE filter is configured from a single prototype pattern (`fit`) and then
detects the same and similar patterns in other images (`transform`). Two
strategies are available: `'Circular Gabor'` and `'Circular DoG'`.
Rotation-, scale- and reflection-invariance are optional.

## Development with uv

The project is managed with [uv](https://docs.astral.sh/uv/).

```shell
git clone https://github.com/mcasl/COSFIRE
cd COSFIRE
uv sync              # creates .venv and installs the locked dependencies
uv run pytest        # run the tests
uv run ruff check cosfire examples
uv run python examples/plot_cosfire_step_by_step.py
```

Add or upgrade dependencies with `uv add <package>` / `uv lock --upgrade`, and
commit the resulting `pyproject.toml` and `uv.lock`.

## Usage

```python
import numpy as np
from cosfire import Cosfire
from cosfire.base import GaborParameters, π

filter_ = Cosfire(
    strategy_name='Circular Gabor',
    center_x=100, center_y=100,            # row and column of the prototype's center
    rho_list=range(0, 100, 10),
    t1=0.99, t2=0.75, t3=0.9,
    filter_parameters=GaborParameters(
        ksize=(10, 10), σ=5,
        θ=np.linspace(0, π, 30, endpoint=False),
        λ=np.linspace(7, 8, 10, endpoint=False),
        γ=0.5, ψ=π,
    ),
    sigma0=0.67, alpha=0.04,
)
filter_.fit(prototype_image)               # configuration
response = filter_.transform(test_image)   # application
```

`rotation_invariant` (default `[0]`), `scale_invariant` (default `[1]`) and
`reflection_invariant` (default `False`) control the invariances.

See `examples/` for a step-by-step demo.

## License

See [LICENSE](LICENSE).
