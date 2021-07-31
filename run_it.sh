#!/bin/sh
storage=/media/akgrown/SG/5GVideoTests
tag=Scenario-3
docker run -it \
    --mount type=bind,source=$storage,target=/data \
    anchortools:$tag python3 $@