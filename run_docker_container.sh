#!/bin/bash
docker run --ipc=host --rm --gpus device=0 --shm-size=64g -itd \
    -v /home/$USER:/home/$USER \
    -v /hdd:/hdd \
    efficient-guided-gen