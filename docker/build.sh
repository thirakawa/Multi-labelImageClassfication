#!/bin/bash


### add exec permission to entrypoint.sh
chmod a+x entrypoint.sh


### build
docker build --tag=hirakawat/torch:2.1.0 --force-rm=true --file=./Dockerfile_torch_2_1_0 .


echo "Build docker; done."
