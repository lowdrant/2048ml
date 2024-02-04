#!/usr/bin/env bash
set -o errexit
virtualenv .venv
source .venv/bin/activate
pip3 install torch
pip3 install numpy
