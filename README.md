# Thrive UI Streamlit App

## Install Docker

[Docker Desktop](https://www.docker.com/products/docker-desktop/)

Complete the full setup/install (requires restart and system admin)

**You can skip the account setup stuff**

## Install UV

Mac OS/Linux - Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows - Install uv

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Configure Application

Copy the .streamlit/secrets_example.toml and Paste it to .streamlit/secrets.toml

Modify the values in the secrets.toml to connect to your desired configuration.

You can configure secrets in `.streamlit/secrets.toml` and access them in your app using `st.secrets.get(...)`.

Cloud based configuration:
set vanna_api, vanna_model

or set vanna_api, vanna_model, anthropic_api, anthropic_model

Hybrid configuration:
set anthropic_api, anthropic_model, chroma_path

or set ollama_host, ollama_model, vanna_api, vanna_model

\*\*Local based configuration:
set ollama_host, ollama_model, chroma_path

`utils/config/training_data.json` Here you can configure your custom training data per your database. No need to pass in DDL, the application will automatically read the DDL and populate that on its own.

`utils/config/forbidden_references.json` Here you can specify any table or column names you want to actively block from querying or training on.

## Start the PostgreSQL Docker Container

```bash
docker compose up -d
```

**restart the powershell**

## Run Streamlit

```bash
uv run streamlit run app.py
```

## Database

Upon starting the docker container, the database and data will be initialized with the following tables:

- `public.penguins`
- `public.titanic_train`
- `public.wny_health`

## Using UV

To get help with uv, run:

```bash
uv help
```

### Init a new project

```bash
uv init my-project
```

or within an existing directory:

```bash
uv init .
```

### Update the projects environment

```bash
uv sync
```

### Add dependencies

```bash
uv add package
```

### Add development dependancies

```bash
uv add package --dev
```

### Create a virtual environment

Note: This will automatically be created with uv run and uv sync

```bash
uv venv
```