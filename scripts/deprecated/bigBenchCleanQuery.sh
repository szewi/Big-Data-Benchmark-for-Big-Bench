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

SETUP_PARAMS="`dirname $0`/bigBenchSetupQueryParams.sh"
if [ ! -f "$SETUP_PARAMS" ]
then
  echo "Query parameter setup file $SETUP_PARAMS not found"
  exit 1
fi

# write environment information into logfile
logEnvInformation

run_main_method () {
  echo "==============================================="
  echo "Cleaning query  : $QUERY_NAME"
  echo "-----------------------------------------------"
  echo "benchmark phase: $BENCHMARK_PHASE"
  echo "stream number  : $STREAM_NUMBER"
  echo "user parameter file : $USER_QUERY_PARAMS_FILE"
  echo "user settings file  :$USER_HIVE_SETTINGS_FILE"
  if [[ -n "$DEBUG_QUERY_PART" ]]; then
    echo "query part to debug: $DEBUG_QUERY_PART"
  fi
  echo "log: $LOG_FILE_NAME"
  echo "==============================================="  

  ### Checking required folder: logs/; tmp/; result/ if they exist, create them and set permissions

  # Run the clean method implemented in the query's run.sh
  "$QUERY_CLEAN_METHOD"

  echo "cleaning dir $RESULT_DIR"
  hadoop fs -rm -r -skipTrash "$RESULT_DIR" &

  echo "cleaning dir $TEMP_DIR"
  hadoop fs -rm -r -skipTrash "$TEMP_DIR" &

  wait

  echo "cleaning log $LOG_FILE_NAME"
  rm -rf "$LOG_FILE_NAME"
}

source "$SETUP_PARAMS"
