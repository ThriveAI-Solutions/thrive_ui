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

Cloud based configuration:
set vanna_api, vanna_model

or set vanna_api, vanna_model, anthropic_api, anthropic_model

Hybrid configuration:
set anthropic_api, anthropic_model, chroma_path

or set ollama_host, ollama_model, vanna_api, vanna_model

**Local based configuration:
set ollama_host, ollama_model, chroma_path

`utils/config/training_data.json` Here you can configure your custom training data per your database.  No need to pass in DDL, the application will automatically read the DDL and populate that on its own.

`utils/config/forbidden_references.json` Here you can specify any table or column names you want to actively block from querying or training on.

# Run

```bash
streamlit run app.py
```

# Database
Run the database create statements int he pgDatabase folder.  This assumes you are using PostgreSQL
