#!/bin/bash
set -e

cd /tts/apex
pip3 install --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' .
cd ..
source ~/ros2_ws/install/setup.sh
python3 TTS_srv.py&

exec "$@"

