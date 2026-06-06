"""test_api_dataframe.py
Verify the DataFrame-first contract for run_pipeline_api and CLI wrapper behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from curve_curator import data_parser, run_pipeline_api
from curve_curator import toml_parser

from tests.integration.test_api_parity import (
    DOSES_UM,
    N_CURVES,
    N_DOSES,
    _make_config,
    _make_synthetic_tsv,
)


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("dataframe_api")
    tsv_path = tmp / "data.tsv"
    _make_synthetic_tsv(tsv_path)
    cfg = toml_parser.set_default_values(_make_config(tsv_path, tmp))
    return {"config": cfg, "tsv_path": tsv_path}


KEY_OUTPUT_COLUMNS = ["Name", "pEC50", "Curve R2", "Curve AUC", "Curve Regulation"]


class TestDataFrameFirstApi:
    def test_disk_loaded_input_matches_in_memory_api_run(self, synthetic_data: dict) -> None:
        """Passing the loaded table in-memory must match the disk-load + API path."""
        config = synthetic_data["config"]
        loaded_df = data_parser.load(config)

        disk_then_api = run_pipeline_api(config, loaded_df, device="cpu")
        in_memory_api = run_pipeline_api(config, loaded_df.copy(), device="cpu")

        assert len(disk_then_api) == len(in_memory_api)
        for column in KEY_OUTPUT_COLUMNS:
            pd.testing.assert_series_equal(
                disk_then_api[column],
                in_memory_api[column],
                rtol=1e-5,
                atol=1e-5,
                check_names=True,
            )

    def test_run_pipeline_api_does_not_load_from_disk(self, synthetic_data: dict) -> None:
        config = synthetic_data["config"]
        input_data = data_parser.load(config)

        with patch("curve_curator.data_parser.load") as mock_load:
            run_pipeline_api(config, input_data, device="cpu")
            mock_load.assert_not_called()

    def test_run_pipeline_api_copies_input_dataframe(self, synthetic_data: dict) -> None:
        config = synthetic_data["config"]
        input_data = data_parser.load(config)
        before = input_data.copy(deep=True)

        run_pipeline_api(config, input_data, device="cpu")

        pd.testing.assert_frame_equal(input_data, before)


def test_cli_calls_run_pipeline_api_with_loaded_data(
    synthetic_data: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = synthetic_data["config"]
    tsv_path = synthetic_data["tsv_path"]
    config_path = tmp_path / "config.toml"
    experiments = [int(x) for x in config["Experiment"]["experiments"]]
    control_experiment = [int(x) for x in config["Experiment"]["control_experiment"]]
    doses = [float(x) for x in config["Experiment"]["doses"]]

    config_path.write_text(
        "\n".join(
            [
                "[Meta]",
                f'id = "{config["Meta"]["id"]}"',
                f'description = "{config["Meta"]["description"]}"',
                'condition = ""',
                f'treatment_time = "{config["Meta"]["treatment_time"]}"',
                "",
                "[Experiment]",
                f"experiments = {experiments}",
                f"control_experiment = {control_experiment}",
                f"doses = {doses}",
                f'dose_scale = "{config["Experiment"]["dose_scale"]}"',
                'dose_unit = "M"',
                'measurement_type = "OTHER"',
                'data_type = "OTHER"',
                'search_engine = "OTHER"',
                'search_engine_version = "0"',
                "",
                "[Paths]",
                f'input_file = "{tsv_path}"',
                f'curves_file = "{tmp_path / "curves.tsv"}"',
                f'normalization_file = "{tmp_path / "norm.txt"}"',
                "",
                '["F Statistic"]',
                f"alpha = {config['F Statistic']['alpha']}",
                f"fc_lim = {config['F Statistic']['fc_lim']}",
            ]
        )
    )

    captured: dict[str, object] = {}

    def _fake_run_pipeline_api(cfg: dict, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        captured["config"] = cfg
        captured["data"] = data
        captured["kwargs"] = kwargs
        return data

    from curve_curator.__main__ import main

    monkeypatch.setattr("curve_curator.__main__.run_pipeline_api", _fake_run_pipeline_api)
    monkeypatch.setattr(sys, "argv", ["curve_curator", str(config_path)])

    with patch("curve_curator.__main__.dashboard.render"):
        main()

    assert "data" in captured
    assert isinstance(captured["data"], pd.DataFrame)
    assert len(captured["data"]) == N_CURVES
    assert captured["data"].columns[0] == "Name"
    assert len(captured["data"].columns) >= 1 + N_DOSES
    assert captured["kwargs"] == {"mad": False, "device": "cpu"}
