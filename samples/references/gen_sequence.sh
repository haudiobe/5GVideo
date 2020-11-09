#!/bin/sh
ffmpeg -y -f lavfi -i testsrc=size=1280x720:n=3:duration=1:rate=30 -c:v rawvideo -pix_fmt yuv420p ./yuv420_1280x720_8bit_rec709.yuv