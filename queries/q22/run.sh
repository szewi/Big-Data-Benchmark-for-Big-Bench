#!/usr/bin/env bash

TEMP_TABLE1="${TEMP_TABLE}_inner"
TEMP_TABLE2="${TEMP_TABLE}_conditional_ratio"

HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_TABLE1=$TEMP_TABLE1 -hiveconf TEMP_TABLE2=$TEMP_TABLE2"

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

        hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE_SCRIPT"
}

query_run_clean_method () {
	hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -e "DROP TABLE IF EXISTS $TEMP_TABLE1; DROP TABLE IF EXISTS $TEMP_TABLE2; DROP TABLE IF EXISTS $RESULT_TABLE;"
}
