#!/usr/bin/env bash
set -euo pipefail

xmake
PYTHONPATH=python python3 -m unittest discover -s test -v