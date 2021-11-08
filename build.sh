#!/bin/sh
set -e

# opts=--no-cache
opts=''

# build Dockerfile.worker dependencies
if [ "$#" -ne 1 ] || [ "$1" = "base" ]; then
    docker build $opts -t vcc-base:latest -f ./docker/Dockerfile.base ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "HM" ]; then
    docker build $opts -t vcc-hm:latest -f ./docker/Dockerfile.HM ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "JM" ]; then
    docker build $opts -t vcc-jm:latest -f ./docker/Dockerfile.JM ./docker
fi

# if [ "$#" -ne 1 ] || [ "$1" = "VTM" ]; then
#     docker build $opts -t vcc-vtm:latest -f ./docker/Dockerfile.VTM ./docker
# fi

# build Dockerfile.queue & Dockerfile.worker
docker-compose build