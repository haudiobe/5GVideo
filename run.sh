#!/bin/sh
scenario=Scenario-3
anchor_key=S3-A36-265
laptop=/home/akgrown/code/jobs/qcom/5GVideo/v2/5GVideo
hdd=/media/akgrown/NILS/5GVIDEO
# docker run -it anchortools:latest ls /home/deps/vmaf/model
docker run -it \
    --mount type=bind,source=$hdd/Anchors,target=/data/Anchors \
    --mount type=bind,source=$hdd/ReferenceSequences,target=/data/ReferenceSequences \
    anchortools:latest ./cmd.py metrics --scenario_dir /data/Anchors/$scenario -k $anchor_key -y
    # anchortools:latest ./verify.py decoder --scenario_dir /data/Anchors/$scenario -k $anchor_key

# --mount type=bind,source=$hdd/tmp/$scenario,target=/data/tmp \
# --tmp-dir $hdd/tmp