"""
Test module for admin_dashboard.py functions.
Tests the is_admin, run_repair, and git_commit_and_push functions.
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
from pages.admin_dashboard import is_admin, run_repair, git_commit_and_push


class TestAdminDashboard:
    """Test cases for admin dashboard functions."""
    
    def test_is_admin_with_correct_password(self):
        """Test is_admin function with correct password."""
        result = is_admin("4250Galt")
        assert result is True
    
    def test_is_admin_with_incorrect_password(self):
        """Test is_admin function with incorrect password."""
        result = is_admin("wrong_password")
        assert result is False
    
    def test_is_admin_with_empty_password(self):
        """Test is_admin function with empty password."""
        result = is_admin("")
        assert result is False

    @patch('pages.admin_dashboard.subprocess.run')
    def test_run_repair_success(self, mock_subprocess):
        """Test run_repair function with successful execution."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.stdout = "Repair completed successfully"
        mock_subprocess.return_value = mock_result
        
        result = run_repair()
        assert result == "Repair completed successfully"
        mock_subprocess.assert_called_once_with(
            ["python", "auto_debug_repair.py"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch('pages.admin_dashboard.subprocess.run')
    def test_run_repair_failure(self, mock_subprocess):
        """Test run_repair function with failed execution."""
        # Mock failed subprocess execution
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, "cmd", stderr="Error occurred")
        
        result = run_repair()
        assert "Repair script failed with error:" in result
        assert "Error occurred" in result

    @patch('pages.admin_dashboard.subprocess.run')
    def test_git_commit_and_push_success(self, mock_subprocess):
        """Test git_commit_and_push function with successful execution."""
        # Mock successful git commands
        mock_subprocess.return_value = Mock()
        
        result = git_commit_and_push()
        assert result is True
        
        # Verify all three git commands were called
        assert mock_subprocess.call_count == 3
        calls = mock_subprocess.call_args_list
        
        # Check git add call
        assert calls[0][0][0] == ["git", "add", "."]
        
        # Check git commit call
        assert calls[1][0][0] == ["git", "commit", "-m", "Automated debug repairs"]
        
        # Check git push call
        assert calls[2][0][0] == ["git", "push"]

    @patch('pages.admin_dashboard.subprocess.run')
    def test_git_commit_and_push_failure(self, mock_subprocess):
        """Test git_commit_and_push function with failed execution."""
        from subprocess import CalledProcessError
        mock_subprocess.side_effect = CalledProcessError(1, "cmd", stderr="Git error")
        
        # Need to mock streamlit as well since it's used in the error handling
        with patch('pages.admin_dashboard.st') as mock_st:
            result = git_commit_and_push()
            assert result is False
            mock_st.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])