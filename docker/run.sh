#!/bin/bash

sudo docker run --rm --runtime=nvidia -v ../.:/tts --network=host --name="soncreo" --cidfile="cidfile" --device /dev/snd -i -t soncreo_cu10:v1 bash 
