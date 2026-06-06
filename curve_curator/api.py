# api.py
# Python API for running the CurveCurator pipeline in-process.
#
# Florian P. Bayer / drevalpy - 2025
#

import os

import pandas as pd

from . import quality_control, quantification, thresholding, toml_parser, torch_fitting


def run_pipeline_api(
    config: dict,
    data: pd.DataFrame,
    *,
    mad: bool = False,
    device: str = "cpu",
    gpu_chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Run the CurveCurator pipeline in-process from a config dict and input table.

    Uses a batched PyTorch fitting backend that runs on CPU or GPU.

    The Python API accepts an already loaded input table. Path-based loading
    belongs to the CLI or other wrappers that call ``data_parser.load`` before
    invoking this function.

    The config dict must satisfy two requirements before being passed here:

    1. All values in ``config['Paths']`` must be **absolute paths** (so that
       ``toml_parser.set_default_values`` → ``update_toml_paths`` is a no-op).
    2. A ``'__file__'`` key must be present:
       ``config['__file__'] = {'Path': '/abs/path/to/config.toml'}``

    CurveCurator's internal print statements are routed through a NullHandler
    logger (configured in ``user_interface.py``) so no output reaches
    sys.stdout/stderr from worker threads.

    Parameters
    ----------
    config:
        Config dict in CurveCurator TOML structure, with absolute paths and
        ``__file__`` injected (see above).
    data:
        CurveCurator input table with ``Name`` and ``Raw *`` columns, as
        produced by ``data_parser.load`` or an equivalent in-memory builder.
    mad:
        Whether to run the MAD outlier analysis step.  Defaults to ``False``
        because ``mad_analysis`` writes ``mad.txt`` to disk.  Pass ``mad=True``
        only when the output directory is intentionally writable.
    device:
        PyTorch device string for the fitting backend, e.g. ``"cpu"``,
        ``"cuda"``, ``"cuda:0"``, ``"mps"``.  Falls back to CPU automatically
        if the requested device is unavailable.
    gpu_chunk_size:
        Maximum number of curves per GPU sub-batch.  Passed directly to
        ``batch_fit_4pl``.  See ``torch_fitting.batch_fit_4pl`` for details.

    Returns
    -------
    pd.DataFrame
        Fitted curves table in CurveCurator output format.
    """
    os.environ.setdefault("TQDM_DISABLE", "1")
    config = toml_parser.set_default_values(config)
    working = data.copy()
    working, _preprocess_result = quantification._preprocess(working, config)
    working = torch_fitting.batch_fit_4pl(working, config, device=device, gpu_chunk_size=gpu_chunk_size)
    working = thresholding.apply_significance_thresholds(working, config=config)
    if mad:
        quality_control.mad_analysis(working, config=config)
    return working
