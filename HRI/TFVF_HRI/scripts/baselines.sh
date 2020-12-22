#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 python baselines/r2plus1d/train.py \
#                     --group-by Scenario \
#                     --save saved_models/baselines/r2plus1d_scenario \
#                     --bs 8

CUDA_VISIBLE_DEVICES=2 python baselines/r2plus1d/train.py \
                    --save saved_models/baselines/r2plus1d_wae \
                    --group-by WAE_id \
                    --bs 8
