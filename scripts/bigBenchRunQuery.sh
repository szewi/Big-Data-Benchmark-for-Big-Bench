#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

if [ $# -lt 1 ]
then
	echo "parameter missing"
	echo "usage: <queryNumber e.g.: 1  or 22 >  "
	exit 1
fi

if [ $1 -lt 10 ]
then
	QUERY_NAME=q0$1	
else
	QUERY_NAME=q$1	
fi	

LOG_FILE_NAME="$BIG_BENCH_LOGS_DIR/$QUERY_NAME.log"

echo "==============================================="
echo "Running query: $QUERY_NAME"
echo "log: $LOG_FILE_NAME"
echo "==============================================="	

### Checking required folder: logs/; tmp/; result/ if they exist, create them and set permissions 

echo "checking existance of: $BIG_BENCH_LOGS_DIR "
if [ ! -d "$BIG_BENCH_LOGS_DIR" ]; then
	mkdir -p "$BIG_BENCH_LOGS_DIR"
fi

if [ ! -e "$LOG_FILE_NAME" ] ; then
    touch "$LOG_FILE_NAME"
fi

if [ ! -w "$LOG_FILE_NAME" ] ; then
    echo "ERROR: cannot write to: $LOG_FILE_NAME, no permission"
    exit 1
fi

echo "checking existence of: ${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR} "
hadoop fs -mkdir -p "${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}"
hadoop fs -chmod uga+rw "${BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR}"

echo "checking existance of: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR} "
hadoop fs -mkdir -p "${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}"
hadoop fs -chmod uga+rw "${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}"

# start timed execution of query. Stderr is appended to stdout and both are written into logs/q??.log and to console

time ("$BIG_BENCH_QUERIES_DIR/$QUERY_NAME/run.sh" ; echo  "======= $QUERY_NAME  time =========") > >(tee -a "$LOG_FILE_NAME") 2>&1 
echo "==========================="

## append query specifc log to allInOne logfile
cat "$LOG_FILE_NAME" >> "$BIG_BENCH_LOGS_DIR/allQueries.log"
