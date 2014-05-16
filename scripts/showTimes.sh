#!/usr/bin/env bash


if [ $# -lt 1 ]
then

	grep -A 10 "time ====" "$BIG_BENCH_LOADING_STAGE_LOG"
	grep -A 10 "time ====" "$BIG_BENCH_LOGS_DIR"/q??.log

else
	if [ $1 -lt 10 ]
	then
		QUERY_NAME=q0$1	
	else
		QUERY_NAME=q$1	
	fi

	grep -A 10 "time ====" "$BIG_BENCH_LOGS_DIR/${QUERY_NAME}.log"

fi



