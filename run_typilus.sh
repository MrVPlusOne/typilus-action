#!/bin/bash
in_dir=$1
out_dir=$2
python="/home/jiayi/anaconda3/envs/typilus-torch/bin/python"

export PYTHONPATH=src
$python -m run_typilus "$in_dir" "$out_dir"