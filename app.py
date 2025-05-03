import streamlit as st
import os
import pandas as pd
import uuid
from datetime import datetime
from resume_processor import load_resume, semantic_chunk_text, store_resume_chunks_in_qdrant, evaluate_resume_for_roles, sanitize_input
from settings import settings

# Set page config as the first Streamlit command
st.set_page_config(page_title="HR Resume Evaluator", layout="wide")

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "job_roles" not in st.session_state:
    st.session_state.job_roles = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File paths
USERS_FILE = os.path.join(settings.FILES_FOLDER, "users.csv")
ROLES_FILE = os.path.join(settings.FILES_FOLDER, "roles.csv")
RESUMES_DIR = os.path.join(settings.FILES_FOLDER, "resumes")

# Ensure directories exist
os.makedirs(settings.FILES_FOLDER, exist_ok=True)
os.makedirs(RESUMES_DIR, exist_ok=True)

# User management
def init_users_file():
    """Initialize users.csv if it doesn't exist."""
    if not os.path.exists(USERS_FILE):
        pd.DataFrame(columns=["username", "password", "email", "full_name", "role", "created_at"]).to_csv(USERS_FILE, index=False, encoding="utf-8")

def read_users():
    """Read users from users.csv with proper encoding."""
    init_users_file()
    try:
        df = pd.read_csv(USERS_FILE, encoding="utf-8", dtype={"username": str, "password": str, "email": str, "full_name": str, "role": str, "created_at": str})
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()
        st.write("Debug: Users read from CSV:", df[["username", "password", "email"]].to_dict())  # Debug log
        return df
    except Exception as e:
        st.error(f"Error reading users: {e}")
        return pd.DataFrame(columns=["username", "password", "email", "full_name", "role", "created_at"])

def save_user(username, password, email, full_name, role):
    """Save a new user to users.csv."""
    try:
        users = read_users()
        # Case-insensitive username check
        if username.lower() in users["username"].str.lower().values:
            return False, "Username already exists."
        new_user = {
            "username": sanitize_input(username),
            "password": sanitize_input(password),  # TODO: Hash in production
            "email": sanitize_input(email),
            "full_name": sanitize_input(full_name) if full_name else "",
            "role": sanitize_input(role) if role else "",
            "created_at": datetime.now().isoformat()
        }
        users = pd.concat([users, pd.DataFrame([new_user])], ignore_index=True)
        users.to_csv(USERS_FILE, index=False, encoding="utf-8")
        st.write("Debug: Saved user:", new_user)  # Debug log
        return True, "User registered successfully!"
    except Exception as e:
        return False, f"Error saving user: {e}"

def validate_login(username, password):
    """Validate user credentials with case-insensitive username."""
    users = read_users()
    username = username.strip()
    password = password.strip()
    st.write(f"Debug: Attempting login with username: {username}, password: {password}")  # Debug log
    user_row = users[(users["username"].str.lower() == username.lower()) & (users["password"] == password)]
    if not user_row.empty:
        st.write("Debug: Login successful for user:", user_row[["username", "password"]].to_dict())  # Debug log
        return True
    st.write("Debug: Login failed, no matching user found.")  # Debug log
    return False

# Role management
def init_roles_file():
    """Initialize roles.csv if it doesn't exist."""
    if not os.path.exists(ROLES_FILE):
        pd.DataFrame(columns=["role_name", "description", "skills", "experience", "education"]).to_csv(ROLES_FILE, index=False, encoding="utf-8")

def load_roles():
    """Load roles from roles.csv into session state."""
    init_roles_file()
    try:
        df = pd.read_csv(ROLES_FILE, encoding="utf-8")
        st.session_state.job_roles = [
            {
                "role_name": row["role_name"],
                "description": row["description"],
                "skills": row["skills"].split(",") if pd.notna(row["skills"]) else [],
                "experience": int(row["experience"]) if pd.notna(row["experience"]) else 0,
                "education": row["education"]
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        st.error(f"Error loading roles: {e}")
        st.session_state.job_roles = []

def save_roles():
    """Save session state roles to roles.csv."""
    try:
        roles_df = pd.DataFrame([
            {
                "role_name": role["role_name"],
                "description": role["description"],
                "skills": ",".join(role["skills"]),
                "experience": role["experience"],
                "education": role["education"]
            }
            for role in st.session_state.job_roles
        ])
        roles_df.to_csv(ROLES_FILE, index=False, encoding="utf-8")
    except Exception as e:
        st.error(f"Error saving roles: {e}")

# File handling
def save_uploaded_file(uploaded_file):
    """Save uploaded resume to resumes directory."""
    try:
        file_path = os.path.join(RESUMES_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# UI pages
def signup_page():
    """Render signup page."""
    st.title("HR Resume Evaluator - Signup")
    with st.form("signup_form"):
        st.header("Create an Account")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email")
        full_name = st.text_input("Full Name (Optional)")
        role = st.text_input("Role (Optional, e.g., HR Manager)")
        submit = st.form_submit_button("Signup")

        if submit:
            if username and password and email:
                success, message = save_user(username, password, email, full_name, role)
                if success:
                    st.success(message)
                    st.session_state.user = username
                    st.session_state.page = "chat"
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please fill in Username, Password, and Email.")
    if st.button("Go to Login"):
        st.session_state.page = "login"
        st.rerun()

def login_page():
    """Render login page."""
    st.title("HR Resume Evaluator - Login")
    with st.form("login_form"):
        st.header("Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username and password:
                if validate_login(username, password):
                    st.session_state.user = username
                    st.session_state.page = "chat"
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            else:
                st.error("Please enter Username and Password.")
    if st.button("Go to Signup"):
        st.session_state.page = "signup"
        st.rerun()

def management_page():
    """Render job role management page."""
    st.title("Job Role Management")
    st.write("Define job roles for resume evaluation.")
    with st.form("job_role_form"):
        st.header("Add New Job Role")
        role_name = st.text_input("Role Name", placeholder="e.g., Systems Support Pharmacist")
        description = st.text_area("Job Description", placeholder="e.g., Looking for a systems support pharmacist...")
        skills = st.text_input("Skills (comma-separated)", placeholder="e.g., Pharma analysis, Excel, MS Office")
        experience = st.number_input("Required Experience (years)", min_value=0, max_value=50, value=0)
        education = st.text_input("Required Education", placeholder="e.g., Bachelor's Degree of Pharmacy")
        submit = st.form_submit_button("Add Role")

        if submit:
            if role_name and description and skills and education:
                role = {
                    "role_name": sanitize_input(role_name),
                    "description": sanitize_input(description),
                    "skills": [sanitize_input(s.strip()) for s in skills.split(",")],
                    "experience": experience,
                    "education": sanitize_input(education)
                }
                st.session_state.job_roles.append(role)
                save_roles()
                st.success(f"Role '{role_name}' added!")
            else:
                st.error("Please fill all fields.")

    st.header("Existing Roles")
    if st.session_state.job_roles:
        for i, role in enumerate(st.session_state.job_roles):
            with st.expander(f"{role['role_name']}"):
                st.write(f"**Description**: {role['description']}")
                st.write(f"**Skills**: {', '.join(role['skills'])}")
                st.write(f"**Experience**: {role['experience']} years")
                st.write(f"**Education**: {role['education']}")
                if st.button("Delete", key=f"delete_{i}"):
                    st.session_state.job_roles.pop(i)
                    save_roles()
                    st.rerun()
    else:
        st.info("No roles defined yet.")

def chat_page():
    """Render resume evaluation chat page."""
    st.title("Resume Evaluation Chat")
    st.write("Upload a resume and select a job role to evaluate eligibility.")

    with st.form("resume_form"):
        st.header("Evaluate a Resume")
        uploaded_file = st.file_uploader("Upload Resume (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
        role_options = [role["role_name"] for role in st.session_state.job_roles]
        selected_role = st.selectbox("Select Job Role", role_options if role_options else ["No roles available"])
        submit = st.form_submit_button("Evaluate Resume")

        if submit:
            if uploaded_file and selected_role != "No roles available":
                with st.spinner("Processing resume..."):
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        try:
                            resume_text = load_resume(file_path)
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": f"Uploaded resume: {uploaded_file.name}",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })

                            chunks = semantic_chunk_text(resume_text)
                            metadata = {
                                "file_name": os.path.basename(file_path),
                                "uploaded_at": datetime.now().isoformat(),
                                "resume_id": str(uuid.uuid4())
                            }
                            vector_store = store_resume_chunks_in_qdrant(chunks, metadata)

                            role = next((r for r in st.session_state.job_roles if r["role_name"] == selected_role), None)
                            if role:
                                results = evaluate_resume_for_roles(resume_text, [role], vector_store)
                                evaluation = results[0]["evaluation"]
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"Evaluation for {selected_role}:\n\n{evaluation}",
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                            else:
                                st.error("Selected role not found.")
                        except Exception as e:
                            st.error(f"Error processing resume: {e}")
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Error: {e}",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
            else:
                st.error("Please upload a resume and select a valid job role.")

    st.header("Chat History")
    for msg in st.session_state.chat_history:
        prefix = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"{prefix} ({msg['timestamp']}): {msg['content']}")

# Main app logic
def main():
    """Run the Streamlit app."""
    load_roles()  # Load roles at startup
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if st.session_state.user is None:
        if st.session_state.page == "signup":
            signup_page()
        else:
            login_page()
    else:
        st.sidebar.title(f"Welcome, {st.session_state.user}")
        page = st.sidebar.radio("Navigate", ["Chat Interface", "Job Role Management", "Logout"])
        if page == "Chat Interface":
            chat_page()
        elif page == "Job Role Management":
            management_page()
        elif page == "Logout":
            st.session_state.user = None
            st.session_state.job_roles = []
            st.session_state.chat_history = []
            st.session_state.page = "login"
            save_roles()
            st.rerun()

if __name__ == "__main__":
    main()