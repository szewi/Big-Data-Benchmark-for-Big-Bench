#!/usr/bin/env bash

if [ $# -lt 1 ]
then
	
	ERRLOG_FILE_NAME="$BIG_BENCH_LOGS_DIR/queryErrors.log"

	grep -n -i -E 'FAIL|ERROR:|Could not|Exception|unexpected' "$BIG_BENCH_LOADING_STAGE_LOG" > $ERRLOG_FILE_NAME
	grep -n -i -E 'FAIL|ERROR:|Could not|Exception|unexpected' "$BIG_BENCH_LOGS_DIR"/q??.log >> $ERRLOG_FILE_NAME

	if [ -s "$ERRLOG_FILE_NAME" ]
	then
		echo "==============================================="
		echo "Errors in queries"
		echo "==============================================="
		cat $ERRLOG_FILE_NAME
	else
		echo "All queries ran successfully"
	fi






else
	if [ $1 -lt 10 ]
	then
	
		QUERY_NAME=q0$1	
	else
		QUERY_NAME=q$1	
	fi
	echo "==============================================="
	echo "Errors in query $QUERY_NAME"
	echo "grep from file:  $BIG_BENCH_LOGS_DIR/$QUERY_NAME.log "
	echo "==============================================="
	grep -n -i -E 'FAIL|ERROR:|Could not|Exception|unexpected'  "$BIG_BENCH_LOGS_DIR/${QUERY_NAME}.log"

fi




