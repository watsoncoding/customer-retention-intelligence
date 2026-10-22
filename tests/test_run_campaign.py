import sys
import pandas as pd
import pytest
from pathlib import Path

from src.retention.run_campaign import main

def test_cli_creates_campaign_csv(monkeypatch, tmp_path):
    output_path = tmp_path / "test_campaign.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_campaign.py",
            "--budget",
            "500",
            "--output",
            str(output_path)
        ]
    )

    main()

    assert output_path.is_file()

    test_campaign = pd.read_csv(output_path)

    assert test_campaign.shape == (25, 5)
    assert test_campaign["customerID"].notna().all()
    assert test_campaign["customerID"].is_unique
    assert (test_campaign["expected_net_value"] > 0).all()

@pytest.mark.parametrize(
    "invalid_budget",
    ["-100", "nan", "inf", "-inf"]
)
def test_cli_rejects_invalid_budget(monkeypatch, invalid_budget):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_campaign.py", "--budget", invalid_budget]
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2

def test_cli_zero_budget(monkeypatch, tmp_path):
    tmp_output_path = tmp_path / "test_campaign.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_campaign.py", "--budget", "0", "--output",  str(tmp_output_path)]
    )
    main()

    assert tmp_output_path.is_file()
    test_campaign_customers = pd.read_csv(tmp_output_path)

    assert test_campaign_customers.shape == (0, 5)
