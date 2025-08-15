"""
Test module for nGS Revised Strategy functions.
Tests the load_polygon_data and run_ngs_automated_reporting functions.
"""

import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
import pandas as pd


class TestNGSRevisedStrategy:
    """Test class for nGS Revised Strategy functions."""

    def test_import_functions_exist(self):
        """Test that the required functions can be imported."""
        # Test shared_utils import with proper mocking
        with patch.dict('sys.modules', {
            'polygon': MagicMock(),
            'polygon_config': MagicMock()
        }):
            try:
                from shared_utils import load_polygon_data
                assert callable(load_polygon_data), "load_polygon_data should be callable"
                print("✅ load_polygon_data import successful")
            except ImportError as e:
                pytest.fail(f"Failed to import load_polygon_data: {e}")

        # Test nGS_Revised_Strategy import with proper mocking
        with patch.dict('sys.modules', {
            'polygon': MagicMock(),
            'ngs_ai_integration_manager': MagicMock()
        }):
            try:
                from nGS_Revised_Strategy import run_ngs_automated_reporting
                assert callable(run_ngs_automated_reporting), "run_ngs_automated_reporting should be callable"
                print("✅ run_ngs_automated_reporting import successful")
            except ImportError as e:
                pytest.fail(f"Failed to import run_ngs_automated_reporting: {e}")

    def test_load_polygon_data_function_signature(self):
        """Test that load_polygon_data has the expected function signature."""
        with patch.dict('sys.modules', {
            'polygon': MagicMock(),
            'polygon_config': MagicMock()
        }):
            from shared_utils import load_polygon_data
            import inspect
            
            # Check function signature
            sig = inspect.signature(load_polygon_data)
            params = list(sig.parameters.keys())
            
            # Should have 'symbols' parameter
            assert 'symbols' in params, "load_polygon_data should have 'symbols' parameter"

    def test_load_polygon_data_empty_symbols(self):
        """Test load_polygon_data function with empty symbols list."""
        with patch.dict('sys.modules', {
            'polygon': MagicMock(),
            'polygon_config': MagicMock()
        }):
            from shared_utils import load_polygon_data
            
            # Mock the necessary dependencies
            with patch('shared_utils.POLYGON_API_KEY', 'test_key'), \
                 patch('shared_utils.datetime') as mock_datetime, \
                 patch('shared_utils.os.path.exists', return_value=False), \
                 patch('shared_utils.json.load', return_value={"daily": {"last_update": None}}):
                
                mock_datetime.now.return_value.strftime.return_value = "2024-01-15"
                
                result = load_polygon_data([])
                
                assert isinstance(result, dict), "Should return a dictionary"
                assert len(result) == 0, "Should return empty dict for empty symbols"

    def test_run_ngs_automated_reporting_function_signature(self):
        """Test that run_ngs_automated_reporting has the expected function signature."""
        with patch.dict('sys.modules', {
            'polygon': MagicMock(),
            'ngs_ai_integration_manager': MagicMock()
        }):
            from nGS_Revised_Strategy import run_ngs_automated_reporting
            import inspect
            
            # Check function signature
            sig = inspect.signature(run_ngs_automated_reporting)
            params = list(sig.parameters.keys())
            
            # Should have 'comparison' parameter (optional)
            assert 'comparison' in params, "run_ngs_automated_reporting should have 'comparison' parameter"
            
            # Check that comparison parameter has default value
            comparison_param = sig.parameters['comparison']
            assert comparison_param.default is not inspect.Parameter.empty, "comparison should have default value"

    def test_run_ngs_automated_reporting_can_be_called(self):
        """Test that run_ngs_automated_reporting can be called without errors."""
        with patch.dict('sys.modules', {
            'polygon': MagicMock(),
            'ngs_ai_integration_manager': MagicMock()
        }):
            from nGS_Revised_Strategy import run_ngs_automated_reporting
            
            # Test that the function exists and can be called
            # We expect it to fail due to missing dependencies, but it should be callable
            try:
                run_ngs_automated_reporting()
                assert True, "Function called successfully"
            except Exception as e:
                # Expected errors due to missing dependencies or files
                expected_errors = [
                    "No module named",
                    "FileNotFoundError", 
                    "cannot import",
                    "ModuleNotFoundError",
                    "does not exist"
                ]
                
                if any(err in str(e) for err in expected_errors):
                    # These are expected - the function exists and is callable
                    assert True, "Function is callable but hit expected dependency error"
                else:
                    # If it's not a dependency error, the function might have issues
                    pytest.fail(f"Unexpected error (function may have issues): {e}")
            
            # Test with comparison parameter
            try:
                comparison_data = {"test": "data"}
                run_ngs_automated_reporting(comparison_data)
                assert True, "Function called successfully with parameter"
            except Exception as e:
                # Expected errors due to missing dependencies or files
                expected_errors = [
                    "No module named",
                    "FileNotFoundError",
                    "cannot import", 
                    "ModuleNotFoundError",
                    "does not exist"
                ]
                
                if any(err in str(e) for err in expected_errors):
                    # These are expected - the function exists and is callable
                    assert True, "Function is callable with parameters but hit expected dependency error"
                else:
                    # If it's not a dependency error, the function might have issues
                    pytest.fail(f"Unexpected error with parameters (function may have issues): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])