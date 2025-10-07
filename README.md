# AI Sentiment-Based Text Generator

Live demo: https://aisentimentgenerator-nssulzrt2sh7rogegretyt.streamlit.app/

Repository: https://github.com/rishavkr43/Ai_sentiment_Generator

## Overview

This project is a Streamlit web app that analyzes the sentiment of an input prompt and generates text aligned with that sentiment. It uses a small sentiment analyzer and a text generation component built with Hugging Face Transformers (or a lightweight generator stub depending on your environment).

Features:
- Automatic or manual sentiment selection
- Display of sentiment scores (optional)
- Generation history saved in session state
- Simple UI with customizable generation options

## Live Link

You can view the deployed live app here:

https://aisentimentgenerator-nssulzrt2sh7rogegretyt.streamlit.app/

## Local Setup

These instructions assume you're on Windows and have Python installed. The project was developed with a virtualenv. Recommended Python version: 3.11 (some packages may not have wheels for 3.13).

1. Clone this repository (if you haven't already):

	git clone https://github.com/rishavkr43/Ai_sentiment_Generator.git
	cd Ai_sentiment_Generator

2. Create a virtual environment (recommended Python 3.11):

	# Use the Python launcher if you have 3.11 installed
	py -3.11 -m venv venv
	.\venv\Scripts\Activate.ps1

	If you don't have `py -3.11`, replace with the path to your Python 3.11 executable.

3. Upgrade pip and install dependencies:

	python -m pip install --upgrade pip
	pip install -r requirements.txt

	Notes:
	- On Windows, some packages (torch, numpy) may require specific wheels. If pip tries to compile from source and fails, consider using Python 3.11 or installing prebuilt wheels (e.g., from the official PyTorch index).

4. Run the Streamlit app locally:

	streamlit run main.py

	The app will open in your browser at `http://localhost:8501` by default.

## Troubleshooting

- Black screen or blank UI:
  - Run `streamlit run main.py` in a terminal to view logs. If a Python exception happens during import or while rendering, Streamlit will show an error in the browser or print a traceback in the terminal.
  - Ensure your Python version is compatible with the dependencies in `requirements.txt`.

- Dependency build errors (e.g., numpy/meson errors):
  - These occur when pip attempts to compile packages from source. Install a supported Python version (3.11) or add the required native build tools (Visual Studio Build Tools) — the easier route is to use Python 3.11.

## Development notes

- The app uses `st.cache_resource` to load models once per session. If you change the model-loading code, restart Streamlit to reload resources.
- Session state is used to store generation history.

## .gitignore

The repository includes a `.gitignore` which excludes `__pycache__/`, `*.pyc`, your virtual environment `venv/`, and common editor files like `.vscode/`.

## License

This project is provided under the MIT license. Use responsibly.

---

If you want, I can also add a short screenshot and a sample prompt/response example to the README. Let me know which you'd prefer.
