#!/bin/bash

export STORAGE_DIR=gs://mcq_storage
export DATA_DIR=${STORAGE_DIR}/datasets
export MODEL_DIR=${STORAGE_DIR}/output_models

git clone https://github.com/oliversssf2/MCQ_generation.git

pip3 install --upgrade pip
pip3 install nltk transformers datasets numpy matplotlib