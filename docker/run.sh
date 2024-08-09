#!/bin/bash


# check command line args
if [ $# -lt 1 ]; then
    echo "ERROR: less arguments"
    echo "USAGE: sh run.sh [container name] [mount volume 1] [mount volume 2] ..."
    exit
fi


# select docker image
imagename="hirakawat/torch:2.1.0"


# port for jupyter
jupyterport="10283"


echo "run docker ..."
echo "    image: ${imagename}"
echo "    container name: ${1}"


# check mount point
if [ $# -gt 1 ]; then
    for var in ${@:2}; do
        mounts+="-v ${var} "
        echo "    mount point: ${var}"
    done
else
    mounts=""
    echo "    mount point: N/A"
fi


# run
docker run --gpus all -ti --rm --ipc=host \
        -u $(id -u):$(id -g) \
        --name=${1} --hostname=${1} \
        -p ${jupyterport}:8888 \
        ${mounts} \
        ${imagename}
