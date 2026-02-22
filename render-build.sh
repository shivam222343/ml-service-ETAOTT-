#!/usr/bin/env bash
# exit on error
set -o errexit

# Set local cargo home for Rust-based builds on Render's read-only system
export CARGO_HOME=$HOME/.cargo
mkdir -p $CARGO_HOME

pip install --upgrade pip
pip install -r requirements.txt

# Install Playwright browser during build phase
python -m playwright install chromium
