#!/bin/sh
run_it(){
    storage=/media/akgrown/SG/5GVideo
    tag=HM-16.22
    docker run -it \
        --mount type=bind,source=$storage,target=/data \
        anchortools:$tag $@
}

run_it download.py  