import pandas as pd
import pytest
from pathlib import Path
from src.create_dataset import get_dataset
#from create_dataset import get_dataset

def test_get_dataset(tmp_path):
    data = {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 31, 35],
        'city': ['New York', 'Los Angeles', 'Chicago'],
    }
    df = pd.DataFrame(data)

    # Save DataFrame to an Excel file
    file_path = tmp_path / "test_file.xlsx"
    df.to_excel(file_path, index=False)

    # Load DataFrame from the file using get_dataset
    loaded_df = get_dataset(file_path)

    # Check that loaded DataFrame equals original DataFrame
    pd.testing.assert_frame_equal(loaded_df, df)

