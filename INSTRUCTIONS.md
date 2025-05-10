## Project Overview
This is an LLM Chatbot Streamlit app. It allows users to ask questions of their database and get sql queries, graphs, and summaries of that data. It uses Vanna as a RAG and to craft a prompt to pass to the LLM. As for it's current configuration, we're using a PostgreSQL database in a docker container with Ollama and ChromaDB.

## Tooling
- `uv`: A Python packager that replaces pip, pyenv, poetry, etc.
- `Vanna`: RAG and intermediary between the LLM and the Streamlit app
- `Streamlit`: Handling the UI
- `pytest`: Testing suite
- `context7`: MCP Server available to the Agent to get the latest documentation for any libraries used

## Additional Notes
If python is required, please prefix with `uv run`. Eg. `uv run streamlit run app.py` or `uv run pytest`

## LLM Agent Instructions
Use TDD (Test Driven Developemnt) to write performant, pythonic, code using software engineering best practices

## TODO
Please mark with an x when complete. Eg. [x]

1.0 - [x] Use `context7` to get the latest Streamlit docs
1.1 - [x] Write a test for the "Generate Graph" button. It seems to be using `get_chart` 
1.2 - [x] Look into why the "Generate Graph" button doesn't seem to be anything when clicked in the UI
1.3 - [x] Write a test to verify this without having to launch the UI and test menually
1.4 - [x] Identify the issue and fix the code
1.5 - [x] Test that the fix is working and adhears to Streamlit, SWE, and Python best practices
1.6 - [] Ask the user to verify the results in the UI