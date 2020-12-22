#!/bin/bash

# NOTE: use `CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/attn_model.sh`
# to train with multiple GPUs
# FLAGS_check_nan_inf=1 python -m paddle.distributed.launch scripts/train_attention_controller.py \
#                    --distributed-training \
#                    --inputs-type inst_crop \
#                    --frame-emb-trainable \
#                    --bs 2 \
#                    --gpu 0 \
#                    --epochs 30 \
#                    --dropout 0.3 \
#                    --run-eval-after-epochs 30 \
#                    --data-worker-gpus-for-train 0 \
#                    --data-worker-gpus-for-test 0 \
#                    --data-workers-for-train 2 \
#                    --data-workers-for-test 1 \
#                    --use-last-act-loss \
#                    --full-neg-train data/full_neg_train_valid_i0.80_s0.10.txt \
#                    --full-neg-test data/full_neg_test_valid_i0.80_s0.10.txt \
#                    --save saved_models/attention/inst_crop

FLAGS_check_nan_inf=1 python scripts/train_attention_controller.py \
                   --inputs_type visual_token \
                   --frame_emb_trainable \
                   --use_last_act_loss \
                   --gpu 1 \
                   --epochs 80 \
                   --run_eval_after_epochs 8 \
                   --data_worker_gpus_for_train 2 \
                   --data_worker_gpus_for_test 2 \
                   --use_decord \
                   --decord_readers 1 \
                   --decord_detectors 2 \
                   --decord_post_workers 4 \
                   --decord_ds_pkl data/decord.pkl \
                   --save saved_models/tmp

# python scripts/save_infer_model_params.py \
#        saved_models/attention/de_inst_crop/epoch_15,saved_models/salutation/test/epoch_19 \
#        jetson/attn_inst_crop \
#        data/raw_wae/wae_lst.pkl \
#        inst_crop
