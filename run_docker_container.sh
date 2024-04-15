#!/bin/bash
docker run --ipc=host --rm --gpus all  -it \
    -v $(pwd):/workspace mpgd