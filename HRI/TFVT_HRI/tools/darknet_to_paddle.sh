#!/bin/bash

cd $(dirname $0)

python darknet_to_keras.py \
       yolov4.cfg \
       ../pretrain_models/yolov4.weights \
       yolov4.h5

python keras_to_tensorflow.py \
       --input_model yolov4.h5 \
       --output_model yolov4.pb

python -m x2paddle.convert \
        --framework tensorflow \
        --model yolov4.pb \
        --save_dir yolov4_paddle \
        --without_data_format_optimization
