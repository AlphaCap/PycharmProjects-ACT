import streamlit as st
import subprocess

# Example: Simple admin authentication (replace with secure method in production)
def is_admin():
    return st.text_input("Enter admin password:", type="password") == "your_admin_password"

def run_repair():
    result = subprocess.run(['python', 'auto_debug_repair.py'], capture_output=True, text=True)
    return "Repairs made" in result.stdout

def git_commit_and_push():
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', 'Automated debug repairs'])
    subprocess.run(['git', 'push'])

st.title("Admin Dashboard")

if is_admin():
    if st.button("Run Automated Repair"):
        repairs_made = run_repair()
        if repairs_made:
            if st.button("Repairs made. Commit and push now?"):
                git_commit_and_push()
                st.success("Changes committed and pushed.")
            else:
                st.info("Review repairs before committing.")
        else:
            st.info("No repairs were needed.")
else:
    st.warning("Admin access required.")
