#!/usr/bin/env bash

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

	TEMP_TABLE1="${TABLE_PREFIX}_inner"
	TEMP_TABLE2="${TABLE_PREFIX}_conditional_ratio"

	HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_TABLE1=$TEMP_TABLE1 -hiveconf TEMP_TABLE2=$TEMP_TABLE2"

        hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE_SCRIPT"
}
