# Activate venv and run pytest
.\.venv\Scripts\Activate.ps1
python -m pip install -r dev-requirements.txt
python -m pytest -q
