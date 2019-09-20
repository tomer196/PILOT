#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py --test-name='20/TSP' --TSP --device='cuda' --decimation-rate=4 --batch-size=24 --lr=0.001 --sub-lr=0.1 --num-epochs=120 --TSP-epoch=40 --G-max=40 --S-max=200 --initialization='gaussian' --acc-weight=1e-1 --vel-weight=1e-1
#
CUDA_VISIBLE_DEVICES=0 python3 reconstructe.py --test-name='20/TSP' --device='cuda' --data-split='test' --batch-size=24
#
python3 common/evaluate.py --test-name='20/TSP'