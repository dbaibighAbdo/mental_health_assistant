# Mental Health Assistant

A small conversational mental health assistant project implemented in Python. This repository contains a simple chatbot module, a Streamlit UI to interact with it, and tests.

## Features

- Lightweight chatbot core in `app/chatbot.py` for handling message flow.
- Configurable parameters in `app/config.py`.
- Streamlit-based UI in `app/streamlit_app.py` for interactive use.
- Unit tests in `app/test_chatbot.py` (run with `pytest`).

## Quick Start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies:

- If the project provides a `requirements.txt` file:

```powershell
pip install -r requirements.txt
```

- Or install from the project `pyproject.toml`/editable install if you prefer:

```powershell
pip install -e .
```

3. Run the Streamlit app locally:

```powershell
streamlit run app/streamlit_app.py
```

4. (Optional) Run the CLI or main entry if present:

```powershell
python main.py
```

## Running Tests

Run the unit tests with `pytest` from the repository root:

```powershell
pytest -q
```

## Project Structure

- `main.py` — optional top-level script / entry point.
- `pyproject.toml` — project metadata and dependency configuration.
- `app/`
	- `chatbot.py` — core chatbot logic.
	- `config.py` — configuration values and helpers.
	- `streamlit_app.py` — Streamlit UI to interact with the assistant.
	- `test_chatbot.py` — unit tests for the chatbot.

## Notes & Next Steps

- If you want, I can:
	- Generate a `requirements.txt` from the environment or `pyproject.toml`.
	- Run the test suite and report results.
	- Add usage screenshots or an example conversation to the README.

If you prefer specific wording, branding, or an expanded developer guide, tell me how you'd like it structured.
