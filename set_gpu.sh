#!/bin/bash

if [ ! -n "$1" ] ;then
    echo "You gpu has been set to: $CUDA_VISIBLE_DEVICES"
else
    echo "You set gpu to: $1"
    export CUDA_VISIBLE_DEVICES=$1
fi
