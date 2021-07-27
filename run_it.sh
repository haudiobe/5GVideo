#!/bin/sh
storage=/path/to/5GVideo
docker run -it \
    --mount type=bind,source=$storage,target=/data \
    anchortools:latest python3 $@