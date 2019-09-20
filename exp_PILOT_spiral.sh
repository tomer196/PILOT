#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --test-name='80/spiral' --device='cuda' --decimation-rate=80 --batch-size=24 --lr=0.001 --sub-lr=0.1 --num-epochs=50 --G-max=40 --S-max=200 --acc-weight 1e-1 --vel-weight 1e-1
#
CUDA_VISIBLE_DEVICES=0 python3 reconstructe.py --test-name='80/spiral' --device='cuda' --data-split='test' --batch-size=24
#
python3 common/evaluate.py --test-name='80/spiral'