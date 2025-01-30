# Thrive UI Streamlit App

# Install

```bash
python -m venv venv
venv\Scripts\Activate

pip install -r requirements.txt
```

# Configure
Copy the .streamlit/secrets_example.toml and Paste it to .streamlit/secrets.toml

Modify the values in the secrets.toml to connect to your desired configuration.

You can configure secrets in `.streamlit/secrets.toml` and access them in your app using `st.secrets.get(...)`.

# Run

```bash
streamlit run app.py
```
<<<<<<< HEAD

# Database
Run the database create statements int he pgDatabase folder.  This assumes you are using PostgreSQL
=======
>>>>>>> main
