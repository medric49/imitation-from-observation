#!/bin/bash

module load python
virtualenv --no-download env
source env/bin/activate
pip install -r requirements.txt
