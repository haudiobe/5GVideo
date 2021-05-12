#!/bin/sh

# assuming /path/to/host/data contains content downloaded from https://dash-large-files.akamaized.net/WAVE/3GPP/5GVideo/
root_dir=/path/to/sample
scenario=Scenario-3
anchor_key=S3-A36-265

docker run -d \
    --mount type=bind,source=$root_dir/Anchors,target=/data/Anchors \
    --mount type=bind,source=$root_dir/ReferenceSequences,target=/data/ReferenceSequences \
    anchortools:latest ./verify.py decoder --scenario_dir /data/Anchors/$scenario -k $anchor_key
