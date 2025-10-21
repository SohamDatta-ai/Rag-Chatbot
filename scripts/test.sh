#!/usr/bin/env bash
source .venv/Scripts/activate
python -m pip install -r dev-requirements.txt
python -m pytest -q
