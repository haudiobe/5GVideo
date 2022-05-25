#!/bin/sh
set -e

echo 'this will take a while, go grab a coffee ...'

# opts=--no-cache
opts=''

# build Dockerfile.worker dependencies
if [ "$#" -ne 1 ] || [ "$1" = "base" ]; then
    docker build $opts -t vcc:base -f ./docker/Dockerfile.base ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "HM" ]; then
    docker build $opts -t vcc:hm -f ./docker/Dockerfile.HM ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "JM" ]; then
    docker build $opts -t vcc:jm -f ./docker/Dockerfile.JM ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "VTM" ]; then
    docker build $opts -t vcc:vtm -f ./docker/Dockerfile.VTM ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "ETM" ]; then
    docker build $opts -t vcc:etm -f ./docker/Dockerfile.ETM ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "AOM" ]; then
    docker build $opts -t vcc:aom -f ./docker/Dockerfile.AOM ./docker
fi

if [ "$#" -ne 1 ] || [ "$1" = "worker" ]; then
    docker build $opts -t vcc:worker -f ./docker/Dockerfile.worker ./
fi

docker-compose build