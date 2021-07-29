#!/bin/sh
storage=/media/akgrown/SG/5GVideoTests
docker run -it \
    --mount type=bind,source=$storage,target=/data \
    anchortools:latest python3 $@