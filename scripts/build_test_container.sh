#!/bin/bash
docker build -t test_model -f TestDockerfile .
docker save -o test_model.tar test_model:latest
docker rmi test_model:latest

#docker load -i test_model.tar
