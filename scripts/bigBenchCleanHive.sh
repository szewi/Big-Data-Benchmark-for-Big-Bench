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
	echo "cleaning HIVE TABLES"
	"$BINARY" $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "${BIG_BENCH_HIVE_SCRIPT_DIR}/dropTables.sql"
	rm -rf "$BIG_BENCH_LOADING_STAGE_LOG"
}

source "$SETUP_PARAMS"
