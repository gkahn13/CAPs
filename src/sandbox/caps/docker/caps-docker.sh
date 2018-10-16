#!/bin/bash

cmd=$1
capsdir=`git rev-parse --show-toplevel`

if [ "$cmd" == "build" ]; then

    dockerfile="Dockerfile-carla"
    if [ -z $dockerfile ]; then
        echo "Need to provide a dockerfile to build"
    else
        echo "Building..."
        docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f $dockerfile -t caps .
        echo "Done building"
    fi

elif [ "$cmd" == "start" ]; then

    echo "Starting..."
    docker run \
        -it \
        --runtime=nvidia \
        --user $(id -u):$(id -g) \
        -v "$capsdir":/home/caps-user/caps \
        -v /media:/media \
        -d \
        -v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
        --name caps \
        caps
    echo "Done starting"

elif [ "$cmd" == "ssh" ]; then

    echo "Sshing..."
    docker exec -it caps /bin/bash
    echo "Done sshing"

elif [ "$cmd" == "stop" ]; then

    echo "Stopping..."
    docker container stop caps
    docker rm caps
    echo "Done stopping"

elif [ "$cmd" == "clean" ]; then

    echo "Cleaning..."
    docker rmi -f $(docker images -q --filter "dangling=true")
    echo "Done cleaning"

elif [ "$cmd" == "--help" ]; then

    echo "Valid commands: build, start, ssh, stop, clean"

else
    echo "INVALID"
fi
