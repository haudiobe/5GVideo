#!/bin/sh

# Scenario 5
docker build \
 --build-arg HDRTOOLS_VERSION=tags/v0.23 \
 --build-arg JM_VERSION=7901703651acf5a64df55615d02de261e9c0ee87 \
 --build-arg HM_VERSION=tags/HM-16.22 \
 --build-arg SCM_VERSION=tags/HM-16.21+SCM-8.8 \
 --build-arg VTM_VERSION=tags/VTM-10.0 \
 -t anchortools:Scenario-5 -f ./docker/Dockerfile .

# Scenario 3
docker build \
 --build-arg HDRTOOLS_VERSION=tags/v0.23 \
 --build-arg JM_VERSION=7901703651acf5a64df55615d02de261e9c0ee87 \
 --build-arg HM_VERSION=tags/HM-16.22 \
 --build-arg SCM_VERSION=tags/HM-16.21+SCM-8.8 \
 --build-arg VTM_VERSION=tags/VTM-13.2 \
 -t anchortools:Scenario-3 -f ./docker/Dockerfile .
