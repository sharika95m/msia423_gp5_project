import pytest
import pandas as pd
import src.generate_features as gf

#Testing calculate_norm_range()
def test_calculate_norm_range_with_valid_input():
    """
    Summary: Happy path for testing calculate_norm_range()
    """
    df = pd.DataFrame({
        'IR_max': [5, 10, 15],
        'IR_min': [1, 2, 3],
        'IR_mean': [2, 5, 8]
    })
    # Define the input arguments
    kwargs = {
        'max_col': 'IR_max',
        'min_col': 'IR_min',
        'mean_col': 'IR_mean'
    }
    # Calculate the normalized range
    result = gf.calculate_norm_range(df, kwargs)
    # Assert the expected output
    expected_output = pd.Series([2.0, 1.6, 1.5])
    pd.testing.assert_series_equal(result, expected_output)

def test_calculate_norm_range_with_missing_column():
    """
    Summary: Unhappy path for testing calculate_norm_range()
    """
    df = pd.DataFrame({
        'IR_max': [5, 10, 15],
        'IR_min': [1, 2, 3],
        'IR_mean': [2, 5, 8]
    })
    # Define the input arguments with a missing column
    kwargs = {
        'max_col': 'IR_max',
        'min_col': 'IR_min',
        'mean_col': 'missing_col'
    }
    # Check if function raises the expected KeyError
    with pytest.raises(SystemExit):
        gf.calculate_norm_range(df, kwargs)

def test_calculate_norm_range_with_str():
    """
    Summary: Unhappy path for testing calculate_norm_range()
    """
    # Create a Pandas dataframe for testing
    df = pd.DataFrame({
        'IR_max': [5, 10, 15],
        'IR_min': [1, 2, 3],
        'IR_mean': ['a', 2, 8]
    })
    # Define the input arguments
    kwargs = {
        'max_col': 'IR_max',
        'min_col': 'IR_min',
        'mean_col': 'IR_mean'
    }
    # Check if function raises the expected TypeError
    with pytest.raises(SystemExit):
        gf.calculate_norm_range(df, kwargs)

#Testing for calculate_range()
def test_calculate_range_with_valid_input():
    """
    Summary: Happy path testing for calculate_range()
    """
    df = pd.DataFrame({
        'IR_max': [5, 10, 15],
        'IR_min': [1, 2, 3]
    })
    # Define the input arguments
    kwargs = {
        'max_col': 'IR_max',
        'min_col': 'IR_min'
    }
    # Calculate the normalized range
    result = gf.calculate_range(df, kwargs)
    # Assert the expected output
    expected_output = pd.Series([4, 8, 12])
    pd.testing.assert_series_equal(result, expected_output)

def test_calculate_range_with_missing_column():
    """
    Summary: Unhappy path testing for calculate_range()
    """
    df = pd.DataFrame({
        'IR_max': [5, 10, 15],
        'IR_min': [1, 2, 3]
    })
    # Define the input arguments with a missing column
    kwargs = {
        'max_col': 'IR_max',
        'min_col': 'missing_col'
    }
    # Check if function raises the expected KeyError
    with pytest.raises(KeyError):
        gf.calculate_range(df, kwargs)

def test_calculate_range_with_str():
    """
    Summary: Unhappy path testing for calculate_range()
    """
    df = pd.DataFrame({
        'IR_max': [5, 10, 15],
        'IR_min': ['a', 2, 8]
    })
    # Define the input arguments
    kwargs = {
        'max_col': 'IR_max',
        'min_col': 'IR_min'
    }
    # Check if function raises the expected TypeError
    with pytest.raises(TypeError):
        gf.calculate_range(df, kwargs)

#Testing for calculate_log_transform()
def test_calculate_log_transform_with_valid_input():
    """
    Summary: Happy path for testing calculate_log_transform()
    """
    df = pd.DataFrame({
        'visible_entropy': [1, 2, 3],
        'IR_min': [4, 5, 6]
    })
    # Define the input arguments
    kwargs = {
        'log_entropy': 'visible_entropy'
    }
    # Calculate the normalized range
    result = gf.calculate_log_transform(df, kwargs)

    # Assert the expected output
    expected_output = pd.Series([0.000000, 0.693147, 1.098612], name='visible_entropy')
    pd.testing.assert_series_equal(result, expected_output)

def test_calculate_log_transform_with_missing_column():
    """
    Summary: Unhappy path for testing calculate_log_transform()
    """
    df = pd.DataFrame({
        'visible_entropy': [2, 5, 8]
    })
    # Define the input arguments with a missing column
    kwargs = {
        'log_entropy': 'missing_col'
    }
    # Check if function raises the expected KeyError
    with pytest.raises(SystemExit):
        gf.calculate_log_transform(df, kwargs)

def test_calculate_log_transform_with_str():
    """
    Summary: Unhappy path for testing calculate_log_transform()
    """
    df = pd.DataFrame({
        'visible_entropy': ['a', 2, 8]
    })
    # Define the input arguments
    kwargs = {
        'log_entropy': 'visible_entropy'
    }
    # Check if function raises the expected TypeError
    with pytest.raises(SystemExit):
        gf.calculate_log_transform(df, kwargs)

#Testing for calculate_entropy()
def test_calculate_entropy_with_valid_input():
    """
    Summary: Happy path for testing calculate_entropy()
    """
    df = pd.DataFrame({
        'visible_entropy': [1, 2, 3],
        'visible_contrast': [4, 5, 6]
    })
    # Define the input arguments
    kwargs = {
        'col_a': 'visible_contrast',
        'col_b': 'visible_entropy'
    }
    # Calculate the normalized range
    result = gf.calculate_entropy(df, kwargs)

    # Assert the expected output
    expected_output = pd.Series([4, 10, 18])
    pd.testing.assert_series_equal(result, expected_output)

def test_calculate_entropy_with_missing_column():
    """
    Summary: Unhappy path for testing calculate_log_transform()
    """
    df = pd.DataFrame({
        'visible_entropy': [1, 2, 3],
        'visible_contrast': [4, 5, 6]
    })
    # Define the input arguments with a missing column
    kwargs = {
        'col_a': 'visible_contrast',
        'col_b': 'missing_col'
    }
    # Check if function raises the expected KeyError
    with pytest.raises(SystemExit):
        gf.calculate_entropy(df, kwargs)

def test_calculate_entropy_with_str():
    """
    Summary: Unhappy path for testing calculate_log_transform()
    """
    df = pd.DataFrame({
        'visible_entropy': [1.2, 2, 3],
        'visible_contrast': ['1', 5, 6]
    })
    # Define the input arguments
    kwargs = {
        'col_a': 'visible_contrast',
        'col_b': 'visible_entropy'
    }

    # Check if function raises the expected TypeError
    with pytest.raises(SystemExit):
        gf.calculate_entropy(df, kwargs)
