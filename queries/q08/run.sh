#!/usr/bin/env bash

# define used temp tables
TEMP_TABLE1="${TEMP_TABLE}_DateRange"
TEMP_TABLE2="${TEMP_TABLE}_sales_review"
TEMP_TABLE3="${TEMP_TABLE}_webSales_date"

HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_TABLE1=$TEMP_TABLE1 -hiveconf TEMP_TABLE2=$TEMP_TABLE2 -hiveconf TEMP_TABLE3=$TEMP_TABLE3"

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
	hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -e "DROP VIEW IF EXISTS $TEMP_TABLE1; DROP VIEW IF EXISTS $TEMP_TABLE2; DROP VIEW IF EXISTS $TEMP_TABLE3; DROP TABLE IF EXISTS $RESULT_TABLE;"
}
