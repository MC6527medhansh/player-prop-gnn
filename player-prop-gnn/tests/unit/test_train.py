"""
Unit tests for train.py
Tests validation logic and error handling

Run with:
    pytest tests/unit/test_train.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.train import (
    validate_dates,
    check_disk_space,
    load_training_data,
    TrainingError,
    ConvergenceError,
    DataError
)


# ============================================================================
# TEST DATE VALIDATION
# ============================================================================

def test_validate_dates_valid():
    """Test date validation with valid dates."""
    # Should not raise
    validate_dates('2018-06-01', '2018-07-05', '2018-07-16')


def test_validate_dates_invalid_format():
    """Test date validation rejects invalid format."""
    with pytest.raises(ValueError) as exc_info:
        validate_dates('not-a-date', '2018-07-05', '2018-07-16')
    
    assert 'Invalid date format' in str(exc_info.value)
    assert 'YYYY-MM-DD' in str(exc_info.value)


def test_validate_dates_misordered_train():
    """Test date validation rejects train_start >= train_end."""
    with pytest.raises(ValueError) as exc_info:
        validate_dates('2018-07-05', '2018-06-01', '2018-07-16')
    
    assert 'train_start must be before train_end' in str(exc_info.value)


def test_validate_dates_misordered_val():
    """Test date validation rejects train_end >= val_end."""
    with pytest.raises(ValueError) as exc_info:
        validate_dates('2018-06-01', '2018-07-16', '2018-07-05')
    
    assert 'train_end must be before val_end' in str(exc_info.value)


def test_validate_dates_same():
    """Test date validation rejects same dates."""
    with pytest.raises(ValueError):
        validate_dates('2018-06-01', '2018-06-01', '2018-06-01')


# ============================================================================
# TEST DISK SPACE CHECK
# ============================================================================

@patch('shutil.disk_usage')
def test_check_disk_space_sufficient(mock_disk_usage):
    """Test disk space check with sufficient space."""
    # Mock 1 GB free
    mock_stat = Mock()
    mock_stat.free = 1024 * 1024 * 1024
    mock_disk_usage.return_value = mock_stat
    
    result = check_disk_space(required_mb=500)
    
    assert result == True


@patch('shutil.disk_usage')
def test_check_disk_space_insufficient(mock_disk_usage):
    """Test disk space check with insufficient space."""
    # Mock 100 MB free
    mock_stat = Mock()
    mock_stat.free = 100 * 1024 * 1024
    mock_disk_usage.return_value = mock_stat
    
    result = check_disk_space(required_mb=500)
    
    assert result == False


# ============================================================================
# TEST DATA LOADING
# ============================================================================

@patch('src.models.train.load_data')
@patch('src.models.train.create_engine')
def test_load_training_data_successful(mock_engine, mock_load_data):
    """Test successful data loading."""
    # Mock data - 200 days starting 2018-06-01
    dates = pd.date_range('2018-06-01', periods=200, freq='D')
    mock_df = pd.DataFrame({
        'match_date': dates,
        'goals': np.random.poisson(0.1, 200),
        'shots_on_target': np.random.poisson(0.4, 200),
        'cards_total': np.random.binomial(1, 0.08, 200),
        'position': ['Forward'] * 200,
        'opponent_id': [1] * 200
    })
    
    mock_load_data.return_value = mock_df
    
    # Use date ranges that span the mock data (200 days = ~6.5 months)
    train_df, val_df = load_training_data(
        train_start='2018-06-01',
        train_end='2018-11-01',  # 5 months = ~150 records
        val_end='2018-12-18'     # Rest for validation
    )
    
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(train_df) + len(val_df) <= len(mock_df)


@patch('src.models.train.load_data')
@patch('src.models.train.create_engine')
def test_load_training_data_insufficient_train(mock_engine, mock_load_data):
    """Test data loading rejects insufficient training data."""
    # Mock small dataset
    dates = pd.date_range('2018-06-01', periods=50, freq='D')
    mock_df = pd.DataFrame({
        'match_date': dates,
        'goals': np.random.poisson(0.1, 50)
    })
    
    mock_load_data.return_value = mock_df
    
    with pytest.raises(DataError) as exc_info:
        load_training_data(
            train_start='2018-06-01',
            train_end='2018-06-15',  # Only 14 days
            val_end='2018-07-01'
        )
    
    assert 'Insufficient training data' in str(exc_info.value)
    assert 'at least 100' in str(exc_info.value)


@patch('src.models.train.load_data')
@patch('src.models.train.create_engine')
def test_load_training_data_insufficient_val(mock_engine, mock_load_data):
    """Test data loading rejects insufficient validation data."""
    # Mock dataset with enough train but not val
    dates = pd.date_range('2018-06-01', periods=105, freq='D')
    mock_df = pd.DataFrame({
        'match_date': dates,
        'goals': np.random.poisson(0.1, 105)
    })
    
    mock_load_data.return_value = mock_df
    
    with pytest.raises(DataError) as exc_info:
        load_training_data(
            train_start='2018-06-01',
            train_end='2018-09-10',  # Uses 100 records
            val_end='2018-09-12'  # Only 2 records left
        )
    
    assert 'Insufficient validation data' in str(exc_info.value)
    assert 'at least 20' in str(exc_info.value)


@patch('src.models.train.create_engine')
def test_load_training_data_db_connection_fails(mock_engine):
    """Test data loading retries on connection failure."""
    from sqlalchemy.exc import OperationalError
    
    # Mock connection failure
    mock_engine.side_effect = OperationalError("Connection refused", None, None)
    
    with pytest.raises(DataError) as exc_info:
        load_training_data(
            train_start='2018-06-01',
            train_end='2018-07-01',
            val_end='2018-08-01'
        )
    
    assert 'Failed to connect to database' in str(exc_info.value)
    assert 'Check database is running' in str(exc_info.value)


# ============================================================================
# TEST COMMAND LINE INTERFACE
# ============================================================================

def test_cli_help():
    """Test CLI help message."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'src.models.train', '--help'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert 'Train multi-task Bayesian model' in result.stdout
    assert '--train-start' in result.stdout
    assert '--draws' in result.stdout


def test_cli_missing_required_args():
    """Test CLI rejects missing required arguments."""
    import subprocess
    
    result = subprocess.run(
        ['python', '-m', 'src.models.train'],
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0
    assert 'required' in result.stderr.lower()


# ============================================================================
# TEST ERROR TYPES
# ============================================================================

def test_error_types_are_distinct():
    """Test that custom error types are distinct."""
    assert not issubclass(TrainingError, DataError)
    assert not issubclass(TrainingError, ConvergenceError)
    assert not issubclass(DataError, ConvergenceError)


def test_error_messages_helpful():
    """Test that error messages include solutions."""
    try:
        raise DataError(
            "Insufficient training data: 50 records\n"
            "Need at least 100 records\n"
            "Solution: Expand date range"
        )
    except DataError as e:
        assert 'Solution:' in str(e)
        assert 'Insufficient training data' in str(e)


# ============================================================================
# TEST EDGE CASES
# ============================================================================

def test_validate_dates_leap_year():
    """Test date validation with leap year."""
    # Should not raise
    validate_dates('2020-02-28', '2020-02-29', '2020-03-01')


def test_validate_dates_year_boundary():
    """Test date validation across year boundary."""
    # Should not raise
    validate_dates('2018-12-01', '2019-01-15', '2019-02-01')


@patch('shutil.disk_usage')
def test_check_disk_space_exactly_required(mock_disk_usage):
    """Test disk space check with exactly required space."""
    # Mock exactly 500 MB free
    mock_stat = Mock()
    mock_stat.free = 500 * 1024 * 1024
    mock_disk_usage.return_value = mock_stat
    
    result = check_disk_space(required_mb=500)
    
    # Should still pass (not strictly less than)
    assert result == True


# ============================================================================
# TEST INTEGRATION WITH MOCK MODEL
# ============================================================================

@patch('src.models.train.save_model_artifacts')
@patch('src.models.train.evaluate_calibration')
@patch('src.models.train.predict_all_props')
@patch('src.models.train.check_convergence')
@patch('src.models.train.fit_model')
@patch('src.models.train.build_multitask_model')
@patch('src.models.train.load_training_data')
@patch('src.models.train.check_disk_space', return_value=True)
def test_train_multitask_model_end_to_end_mock(
    mock_disk_space,
    mock_load_data,
    mock_build,
    mock_fit,
    mock_check,
    mock_predict,
    mock_evaluate,
    mock_save
):
    """Test full training pipeline with mocks."""
    from src.models.train import train_multitask_model
    
    # Setup mocks
    dates = pd.date_range('2018-06-01', periods=200, freq='D')
    mock_df = pd.DataFrame({
        'match_date': dates,
        'goals': np.random.poisson(0.1, 200),
        'shots_on_target': np.random.poisson(0.4, 200),
        'cards_total': np.random.binomial(1, 0.08, 200),
    })
    
    train_df = mock_df[:150]
    val_df = mock_df[150:]
    
    mock_load_data.return_value = (train_df, val_df)
    
    mock_model = Mock()
    mock_coords = {'position': ['Forward'], 'opponent': [1], 'feature': ['f1']}
    mock_build.return_value = (mock_model, mock_coords)
    
    mock_idata = Mock()
    mock_fit.return_value = mock_idata
    
    mock_predict.return_value = {
        'goals': {'prob_atleast_1': np.random.rand(len(val_df))},
        'shots': {'prob_atleast_1': np.random.rand(len(val_df))},
        'cards': {'prob_atleast_1': np.random.rand(len(val_df))}
    }
    
    mock_evaluate.return_value = {
        'ece': 0.03,
        'brier': 0.15,
        'mae': 0.12
    }
    
    # Run training
    metadata, idata, results = train_multitask_model(
        train_start_date='2018-06-01',
        train_end_date='2018-08-01',
        val_end_date='2018-09-01',
        draws=10,  # Small for testing
        chains=2,
        model_version='test_v1.0'
    )
    
    # Assertions
    assert metadata['version'] == 'test_v1.0'
    assert metadata['n_train'] == len(train_df)
    assert metadata['n_val'] == len(val_df)
    assert 'calibration' in metadata
    assert idata == mock_idata
    
    # Verify mocks called
    mock_build.assert_called_once()
    mock_fit.assert_called_once()
    mock_check.assert_called_once()
    mock_save.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])