#!/bin/bash

module load python/3.8
cd drddprojenv
source $HOME/drddprojenv/bin/activate
python -m spacy download en_core_web_lg
export quarter=q1
python 'get_roberta_scores.py'