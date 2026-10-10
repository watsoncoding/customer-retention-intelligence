import pandas as pd
import pytest

from src.data.validate_data import validate_tenure

def test_validate_tenure_valid():
    df = pd.DataFrame(
        {
            "tenure": [0, 12, 24, 36]
        }
    )

    assert validate_tenure(df) == True

@pytest.mark.parametrize("value", [-1, 2.5, None, "unknown"])
def test_validate_tenure_invalid(value):
    df = pd.DataFrame(
        {
            "tenure": [value]
        }
    )

    with pytest.raises(ValueError):
        validate_tenure(df)
