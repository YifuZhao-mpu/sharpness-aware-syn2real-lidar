#!/bin/bash
PY=/home/zyf/anaconda3/envs/pointcept/bin/python
export CUDA_VISIBLE_DEVICES=0
$PY bn_recalib.py --runs none_full --verify_before
$PY bn_recalib.py --runs none_s2 none_s3 sam005_full sam005_s2 sam005_s3
