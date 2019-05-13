#!/bin/bash
set -e

cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..

python3 TTS_srv_new.py&

exec "$@"


