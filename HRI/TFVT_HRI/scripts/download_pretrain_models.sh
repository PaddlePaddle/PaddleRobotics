#!/bin/bash

if [ ! -d pretrain_models ]; then
    mkdir pretrain_models && cd pretrain_models
else
    cd pretrain_models
fi

check_wget() {
    local url=$1

    wget_output=$(wget -q "$url")
    if [ $? -ne 0 ]; then
        echo "Failed to download "$url
        exit 1
    else
        echo "Downloaded "$url
    fi
}

# Scene understanding: YOLOv3
check_wget https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar
tar -xvf yolov3_r34.tar

# Scene understanding: YOLOv4
check_wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# ERNIE 1.0
check_wget https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz
mkdir ERNIE_v1
tar -zxvf ERNIE_1.0_max-len-512.tar.gz -C ERNIE_v1

# Feature extraction for tracker, i.e. prepare videos for annotation
check_wget https://raw.githubusercontent.com/Qidian213/deep_sort_yolov3/master/model_data/mars-small128.pb
