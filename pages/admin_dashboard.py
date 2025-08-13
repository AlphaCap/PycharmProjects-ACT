import subprocess
import streamlit as st


# Example: Simple admin authentication (replace with a secure method in production)
def is_admin(password: str) -> bool:
    """
    Check if the given password matches the admin password.
    """
    ADMIN_PASSWORD = "4250Galt"
    return password == ADMIN_PASSWORD

def run_repair() -> str:
    """
    Execute the auto_debug_repair script and return the result.
    """
    try:
        result = subprocess.run(
            ["python", "auto_debug_repair.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Repair script failed with error:\n{e.stderr}"
    except Exception as e:
        return f"Unexpected error occurred: {e}"


def git_commit_and_push() -> bool:
    """
    Commit and push changes to the Git repository.
    """
    try:
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Automated debug repairs"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "push"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Git command failed with error:\n{e.stderr}")
        return False
    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")
        return False


def admin_dashboard():
    """
    Main function to render the Admin Dashboard.
    """
    st.title("Admin Dashboard")

    # Verify admin access
    password = st.text_input("Enter admin password:", type="password")
    if is_admin(password):
        # Perform automated repair action
        if st.button("Run Automated Repair"):
            repair_output = run_repair()
            st.text(f"Repair Output:\n{repair_output}")

            # Prompt for Git commit and push if repairs were made
            if "Repairs made" in repair_output:
                if st.button("Repairs made. Commit and push now?"):
                    if git_commit_and_push():
                        st.success("Changes committed and pushed successfully.")
                    else:
                        st.error("Failed to commit and push changes.")
                else:
                    st.info("Review repairs before committing.")
            else:
                st.warning("No repairs were needed or an error occurred during the repair process.")
    else:
        st.warning("Admin access required.")


if __name__ == "__main__":
    admin_dashboard()
