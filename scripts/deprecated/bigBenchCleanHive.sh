#!/usr/bin/env bash

# Source basic environment
ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
  echo "Environment setup file $ENV_SETTINGS not found"
  exit 1
else
  source "$ENV_SETTINGS"
fi

# write environment information into logfile
logEnvInformation

echo "cleaning HIVE TABLES"
"$HIVE_BINARY" -f "${BIG_BENCH_HIVE_SCRIPT_DIR}/dropTables.sql"

