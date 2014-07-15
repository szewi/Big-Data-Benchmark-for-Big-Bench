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

if [ $# -lt 1 ]
then
	grep -A 10 "time ====" "$BIG_BENCH_LOADING_STAGE_LOG"
	grep -A 10 "time ====" "$BIG_BENCH_LOGS_DIR"/q[0-9][0-9]*.log
else
	if [ $1 -lt 10 ]
	then
		QUERY_NAME=q0$1	
	else
		QUERY_NAME=q$1	
	fi

	grep -A 10 "time ====" "$BIG_BENCH_LOGS_DIR"/${QUERY_NAME}*.log
fi
