#!/bin/bash
# Add project root to PYTHONPATH for package discovery without pip install
export PYTHONPATH="${PIXI_PROJECT_ROOT:-.}:${PYTHONPATH:-}"
