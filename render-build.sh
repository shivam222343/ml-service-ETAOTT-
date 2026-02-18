#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browsers (without system deps that require sudo)
playwright install chromium
