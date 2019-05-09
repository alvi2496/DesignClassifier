#!/usr/bin/env bash

PROJECT_ROOT=`pwd`

cd apps/scraper/

FILENAME=ruby scripts/main.rb $1 $2 ${PROJECT_ROOT}

echo ${FILENAME}

cd ${PROJECT_ROOT}
