#!/usr/bin/env bash

query_run_main_method () {

	HIVE1_SCRIPT="$QUERY_DIR/q23_1.sql"
	if [ ! -r "$HIVE1_SCRIPT" ]
	then
		echo "SQL file $HIVE1_SCRIPT can not be read."
		exit 1
	fi

	HIVE2_SCRIPT="$QUERY_DIR/q23_2.sql"
	if [ ! -r "$HIVE2_SCRIPT" ]
	then
		echo "SQL file $HIVE2_SCRIPT can not be read."
		exit 1
	fi

	HIVE3_SCRIPT="$QUERY_DIR/q23_3.sql"
	if [ ! -r "$HIVE3_SCRIPT" ]
	then
		echo "SQL file $HIVE3_SCRIPT can not be read."
		exit 1
	fi

	RESULT_TABLE1="${RESULT_TABLE}1"
	RESULT_DIR1="$RESULT_DIR/$RESULT_TABLE1"
	RESULT_TABLE2="${RESULT_TABLE}2"
	RESULT_DIR2="$RESULT_DIR/$RESULT_TABLE2"

	HIVE_PARAMS="$HIVE_PARAMS -hiveconf RESULT_TABLE1=$RESULT_TABLE1 -hiveconf RESULT_DIR1=$RESULT_DIR1 -hiveconf RESULT_TABLE2=$RESULT_TABLE2 -hiveconf RESULT_DIR2=$RESULT_DIR2"

	#hadoop fs -rm -r -skipTrash "${RESULT_DIR}"/*
	#hadoop fs -mkdir -p "${RESULT_DIR}"
	#hadoop fs -chmod uga+rw "${RESULT_DIR}"


if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 1/4: make view"
	echo "========================="
	hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE1_SCRIPT"
fi

if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 2 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 2/4: make result 1"
	echo "========================="
	hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE2_SCRIPT"
fi

if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 3 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 3/4: make result 2"
	echo "========================="
	hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE3_SCRIPT"
fi

if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 4 ]] ; then
	echo "========================="
	echo "$QUERY_NAME Step 4/4: cleanup"
	echo "========================="
	hive $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "${QUERY_DIR}/cleanup.sql"
fi
}
