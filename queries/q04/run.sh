#!/usr/bin/env bash

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

	# define used temp tables
	TEMP_TABLE1="${TABLE_PREFIX}_tmp_sessions"
	TEMP_TABLE2="${TABLE_PREFIX}_tmp_cart_abandon"

	HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_TABLE1=$TEMP_TABLE1 -hiveconf TEMP_TABLE2=$TEMP_TABLE2"

        hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE_SCRIPT"

	echo "======= $QUERY_NAME result =======" | tee -a "$LOG_FILE_NAME" 2>&1
	echo "results in: $RESULT_DIR" | tee -a "$LOG_FILE_NAME" 2>&1
	echo "to display: hadoop fs -ls $RESULT_DIR/*" | tee -a "$LOG_FILE_NAME" 2>&1
	echo "=========================" | tee -a "$LOG_FILE_NAME" 2>&1
}
