#!/bin/bash
PY=/home/zyf/anaconda3/envs/pointcept/bin/python
export CUDA_VISIBLE_DEVICES=1
$PY bn_recalib.py --runs sampolar_s1 --verify_before
$PY bn_recalib.py --runs sampolar_s2 sampolar_s3 polarmix_s1 polarmix_s2 polarmix_s3
