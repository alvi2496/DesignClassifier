#!/usr/bin/env bash

PROJECT_ROOT=`pwd`

cd apps/scraper/

ruby scripts/main.rb $1 ${PROJECT_ROOT}

cd ${PROJECT_ROOT}

cd apps/classifier/

source venv/bin/activate

python scripts/main.py ${PROJECT_ROOT}

deactivate

cd ${PROJECT_ROOT}

rm comments.csv
