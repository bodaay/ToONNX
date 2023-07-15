#!/bin/bash

DockerTagName="ghcr.io/bodaay/toonnx:latest"

DockerFileToBuild="Dockerfile"
docker build -f $DockerFileToBuild -t $DockerTagName .