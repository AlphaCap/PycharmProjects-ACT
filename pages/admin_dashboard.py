import subprocess
import streamlit as st

# Example: Simple admin authentication (replace with secure method in production)
def is_admin():
    return st.text_input("Enter admin password:", type="password") == "4250Galt"

def run_repair():
    try:
        result = subprocess.run(
            ["python", "auto_debug_repair.py"], capture_output=True, text=True, check=True
        )
        st.text(f"Repair Output:\n{result.stdout}")
        return "Repairs made" in result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"Repair script failed with error:\n{e.stderr}")
        return False
    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")
        return False

def git_commit_and_push():
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Automated debug repairs"], check=True)
        subprocess.run(["git", "push"], check=True)
        st.success("Changes committed and pushed successfully.")
    except subprocess.CalledProcessError as e:
        st.error(f"Git command failed with error:\n{e.stderr}")
    except Exception as e:
        st.error(f"Unexpected error occurred: {e}")

st.title("Admin Dashboard")

if is_admin():
    if st.button("Run Automated Repair"):
        repairs_made = run_repair()
        if repairs_made:
            if st.button("Repairs made. Commit and push now?"):
                git_commit_and_push()
            else:
                st.info("Review repairs before committing.")
        else:
            st.info("No repairs were needed or an error occurred during the repair process.")
else:
    st.warning("Admin access required.")
