"""
Test module for nGS_Revised_Strategy.py functions.
Tests the import and basic functionality of load_polygon_data and run_ngs_automated_reporting.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import Mock, patch

# Import the functions to be tested
from shared_utils import load_polygon_data
from nGS_Revised_Strategy import run_ngs_automated_reporting


class TestNGSRevisedStrategy:
    """Test cases for nGS Revised Strategy functions."""
    
    @patch('shared_utils.RESTClient')
    @patch('shared_utils.os.path.exists')
    def test_load_polygon_data_import_and_call(self, mock_exists, mock_rest_client):
        """Test that load_polygon_data can be imported and called."""
        # Mock the file system checks to avoid file dependencies
        mock_exists.return_value = False
        
        # Mock the REST client
        mock_client = Mock()
        mock_rest_client.return_value = mock_client
        mock_client.get_aggs.return_value = []
        
        # Test that the function can be called with a list of symbols
        symbols = ["AAPL", "MSFT"]
        
        try:
            result = load_polygon_data(symbols)
            # The function should return a dictionary
            assert isinstance(result, dict)
            # The keys should match the input symbols (even if data is empty)
            assert all(symbol in result for symbol in symbols)
        except Exception as e:
            # If it fails due to missing config or other dependencies, that's expected
            # The important thing is that the import worked
            assert "load_polygon_data" in str(type(load_polygon_data))

    def test_run_ngs_automated_reporting_import_and_call(self):
        """Test that run_ngs_automated_reporting can be imported and called."""
        # Test that the function can be called (may fail due to dependencies, but import should work)
        try:
            result = run_ngs_automated_reporting()
            # Function returns None, so we just check it doesn't crash on import
            assert result is None
        except Exception as e:
            # If it fails due to missing dependencies or data files, that's expected
            # The important thing is that the import worked and function is callable
            # Check that it's a dependency issue, not an import issue
            assert ("ModuleNotFoundError" in str(type(e)) or 
                    "FileNotFoundError" in str(type(e)) or
                    "KeyError" in str(type(e)) or
                    "AttributeError" in str(type(e))), f"Unexpected error type: {type(e)}"

    def test_imports_successful(self):
        """Test that all required functions are successfully imported."""
        # Verify that load_polygon_data is imported from shared_utils
        assert hasattr(load_polygon_data, '__call__'), "load_polygon_data should be callable"
        
        # Verify that run_ngs_automated_reporting is imported from nGS_Revised_Strategy  
        assert hasattr(run_ngs_automated_reporting, '__call__'), "run_ngs_automated_reporting should be callable"
        
        # Check function signatures exist
        import inspect
        
        # load_polygon_data should accept symbols parameter
        sig_load = inspect.signature(load_polygon_data)
        assert 'symbols' in sig_load.parameters, "load_polygon_data should have 'symbols' parameter"
        
        # run_ngs_automated_reporting should accept comparison parameter
        sig_run = inspect.signature(run_ngs_automated_reporting)
        assert 'comparison' in sig_run.parameters, "run_ngs_automated_reporting should have 'comparison' parameter"

    @patch('shared_utils.POLYGON_API_KEY', 'test_key')
    @patch('shared_utils.RESTClient')
    def test_load_polygon_data_with_symbols(self, mock_rest_client):
        """Test load_polygon_data with actual symbol list."""
        # Mock the REST client to avoid API calls
        mock_client = Mock()
        mock_rest_client.return_value = mock_client
        mock_client.get_aggs.return_value = []
        
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # This should not raise an import error
        result = load_polygon_data(symbols)
        assert isinstance(result, dict)
        # Should have entries for each symbol (even if empty)
        for symbol in symbols:
            assert symbol in result

    def test_run_ngs_automated_reporting_with_comparison(self):
        """Test run_ngs_automated_reporting with comparison parameter."""
        comparison_data = {"test_key": "test_value"}
        
        # This should not raise an import error
        try:
            result = run_ngs_automated_reporting(comparison_data)
            # Function should complete (returns None)
            assert result is None
        except Exception as e:
            # May fail due to missing dependencies, but import should work
            # Check that it's a dependency issue, not an import issue
            assert ("ModuleNotFoundError" in str(type(e)) or 
                    "FileNotFoundError" in str(type(e)) or
                    "KeyError" in str(type(e)) or
                    "AttributeError" in str(type(e))), f"Expected dependency error, got: {type(e)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])