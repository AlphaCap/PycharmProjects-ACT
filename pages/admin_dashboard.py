import os
import subprocess
import sys

# Make Streamlit optional so tests can import this module without the package installed
try:
    import streamlit as st  # type: ignore
except Exception:
    class _DummySt:
        def error(self, *args, **kwargs):
            print(*args)

        def info(self, *args, **kwargs):
            print(*args)

        def warning(self, *args, **kwargs):
            print(*args)

        def success(self, *args, **kwargs):
            print(*args)

        # UI-only methods used in admin_dashboard(); harmless no-ops under tests
        def title(self, *args, **kwargs):
            pass

        def text(self, *args, **kwargs):
            pass

        def button(self, *args, **kwargs):
            return False

        def checkbox(self, *args, **kwargs):
            return False

    st = _DummySt()


# Simple admin authentication (hard-coded by requirement)
def is_admin(password: str | None = None) -> bool:
    """
    Check if the given password matches the admin password.
    If password is None (e.g., some tests call is_admin() without args), return False.
    """
    ADMIN_PASSWORD = "4250Galt"
    if password is None:
        return False
    return password == ADMIN_PASSWORD


def run_repair() -> str:
    """
    Execute the auto_debug_repair script and return the result.
    Uses the current Python interpreter to avoid PATH issues.
    """
    try:
        result = subprocess.run(
            [sys.executable, "auto_debug_repair.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Repair script failed with error:\n{e.stderr}"
    except Exception as e:
        return f"Unexpected error occurred: {e}"


def git_commit_and_push(dry_run: bool | None = None) -> bool:
    """
    Commit and push changes to the Git repository.
    - If dry_run is True or env var NGS_DRY_RUN=1 is set, simulate success (no network).
    - Gracefully handle common git states (nothing to commit, no upstream).
    """
    try:
        # Dry run mode for tests/CI
        if dry_run is True or os.getenv("NGS_DRY_RUN") == "1":
            # Simulate the flow and success for tests (no git side-effects)
            print("NGS_DRY_RUN enabled: simulating git add/commit/push success")
            return True

        # Stage changes
        add_proc = subprocess.run(["git", "add", "."], capture_output=True, text=True)
        if add_proc.returncode != 0:
            st.error(f"git add failed:\n{add_proc.stderr}")
            return False

        # Commit changes (allow no-op if nothing to commit)
        commit_proc = subprocess.run(
            ["git", "commit", "-m", "Automated debug repairs"],
            capture_output=True,
            text=True,
        )
        if commit_proc.returncode != 0:
            # Common case: nothing to commit
            if (
                "nothing to commit" in (commit_proc.stderr or "").lower()
                or "nothing to commit" in (commit_proc.stdout or "").lower()
            ):
                st.info("No changes to commit. Skipping push.")
                return True
            st.error(f"git commit failed:\n{commit_proc.stderr or commit_proc.stdout}")
            return False

        # Push changes (handle missing upstream)
        push_proc = subprocess.run(["git", "push"], capture_output=True, text=True)
        if push_proc.returncode != 0:
            # Detect missing upstream and provide actionable message
            if "has no upstream branch" in (push_proc.stderr or "").lower():
                st.error(
                    "Push failed: No upstream branch configured. "
                    "Set the upstream, e.g.: git push --set-upstream origin <branch>"
                )
            else:
                st.error(f"git push failed:\n{push_proc.stderr or push_proc.stdout}")
            return False

        return True

    except Exception as e:
        # Avoid hard failure if Streamlit isn't available (e.g., under tests)
        try:
            st.error(f"Unexpected error occurred: {e}")
        except Exception:
            print(f"Unexpected error occurred: {e}")
        return False


def admin_dashboard():
    """
    Main function to render the Admin Dashboard.
    """
    st.title("Admin Dashboard")

    # Verify admin access
    password = st.text_input("Enter admin password:", type="password") if hasattr(st, "text_input") else None
    if is_admin(password):
        # Perform automated repair action
        if st.button("Run Automated Repair"):
            repair_output = run_repair()
            st.text(f"Repair Output:\n{repair_output}")

            # Prompt for Git commit and push if repairs were made
            if "Repairs made" in repair_output:
                st.warning(
                    "Repairs were made. Please confirm you authorize committing and pushing these changes."
                )

                # Explicit authorization checkbox
                approved = st.checkbox(
                    "I authorize commit and push to the remote repository"
                )

                # Only show/enable the push button when approved
                if st.button("Commit and Push Now", disabled=not approved):
                    # Normal operation (not dry-run) in UI
                    if git_commit_and_push(dry_run=False):
                        st.success("Changes committed and pushed successfully.")
                    else:
                        st.error("Failed to commit and/or push changes.")
                else:
                    st.info("Review the repair output and approve before pushing.")
            else:
                st.info(
                    "No repairs were needed or the repair process reported no changes."
                )
    else:
        st.warning("Admin access required.")


if __name__ == "__main__":
    admin_dashboard()
