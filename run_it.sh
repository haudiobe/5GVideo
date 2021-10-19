#!/bin/sh
storage=/path/to/5GVideo
tag=Scenario-3
docker run -it \
    --mount type=bind,source=$storage,target=/data \
    anchortools:$tag python3 $@