#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py -lr 0.0001 -nw 0 -cp ../model_checkpoints_r50
