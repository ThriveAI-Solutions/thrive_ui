# Thrive UI Streamlit App

# Install

```bash
python3 -m venv venv
source venv/bin/activate

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

# Database
Run the database create statements int he pgDatabase folder.  This assumes you are using PostgreSQL