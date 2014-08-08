#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

logEnvInformation

if [ -d "$BIG_BENCH_LOGS_DIR" ]
then
  cd "$BIG_BENCH_LOGS_DIR"
  zip -r logs-`date +%Y%m%d-%H%M%S`.zip *.log *.csv
fi
