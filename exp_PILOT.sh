#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --test-name = 'PILOT-TSP' \
                --data-parallel = False \
                --device = 'cuda' \
                --decimation-rate = 20 \
                --batch-size = 16 \
                --lr = 0.001 \
                --sub-lr = 0.1 \
                --num-epochs = 40 \
                --TSP-epoch = 30 \
                --trajectory-learning = True \
                --a-max = 3 \
                --v-max = 30 \
                --initialization = 'gaussian'

CUDA_VISIBLE_DEVICES=0 python reconstructe.py --test-name = 'PILOT-TSP' \
                --device = 'cuda' \
                --data-split = 'val' \
                --batch-size = 16 \

CUDA_VISIBLE_DEVICES=0 python common.evaluate.py --test-name = 'PILOT-TSP' \
                --data-split='val' \

exit