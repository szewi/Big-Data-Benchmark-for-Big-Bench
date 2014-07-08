#!/usr/bin/env bash

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

	#EXECUTION Plan:
	#step 1.  rm/mkdir TEMP_DIR	:	hadoop fs rm/mkdir TEMP_DIR
	#step 2.  hive q05.sql		:	Run hive querys to extract the input data
	#step 3.  mahout TrainLogistic	:	Train logistic regression model
	#step 4.  mahout calc log_reg 	:	Calculating logistic regression for input data
	#step 5.  mahout dump > hdfs/res:	Converting result and copy result do hdfs query result folder
	#step 6.  hive && hdfs 		:	cleanup.sql && hadoop fs rm MH

	HDFS_RESULT_FILE="${RESULT_DIR}/logRegResult.txt"

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 1/4: Prepare temp dir"
		echo "========================="
		#hadoop fs -rm -r "$TEMP_DIR" &
		#hadoop fs -rm -r "$RESULT_DIR" &
		#wait
		#hadoop fs -mkdir -p "$TEMP_DIR" &
		#hadoop fs -mkdir -p "$RESULT_DIR" &
		#wait
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 2 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 2/4: Executing hive queries"
		echo "tmp output: ${TEMP_DIR}"
		echo "========================="
		# Write input for k-means into ctable
		"$BINARY" $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "$HIVE_SCRIPT"
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 3 ]] ; then
	
		echo "========================="
		echo "$QUERY_NAME Step 3/4: log regression"
		echo "========================="

		TMP_LOG_REG_IN_FILE="`mktemp`"
		TMP_LOG_REG_MODEL_FILE="`mktemp`"

		echo "-------------------------"
		echo "$QUERY_NAME Step 3/4 Part 1: Copy hive result to local csv file"
		echo "tmp output: ${TMP_LOG_REG_IN_FILE}"
		echo "-------------------------"

		echo "streaming result from hive ..."
		#write header
		echo '"c_customer_sk","college_education","male","label"' > "${TMP_LOG_REG_IN_FILE}"
		# append hive result
		hadoop fs -cat "${TEMP_DIR}"/* >> "${TMP_LOG_REG_IN_FILE}"
		echo "streaming result from hive ... done"
		echo "sample:"
		echo "size: " `du -bh "${TMP_LOG_REG_IN_FILE}"`
		echo "------"
		head "${TMP_LOG_REG_IN_FILE}"
		echo "..." 
		echo "-------------------------"

		echo "$QUERY_NAME Step 3/4 Part 2: Train logistic model"
		echo "Command " mahout trainlogistic --input "$TMP_LOG_REG_IN_FILE" --output "$TMP_LOG_REG_MODEL_FILE" --target c_customer_sk --categories 2 --predictors college_education male label --types n n n --passes 20 --features 20 --rate 1 --lambda 0.5
		echo "tmp output: ${TMP_LOG_REG_MODEL_FILE}"
		echo "-------------------------"
	
		mahout trainlogistic --input "$TMP_LOG_REG_IN_FILE" --output "$TMP_LOG_REG_MODEL_FILE" --target c_customer_sk --categories 2 --predictors college_education male label --types n n n  --passes 20 --features 20 --rate 1 --lambda 0.5	

		echo "-------------------------"
		echo "$QUERY_NAME Step 3/4 Part 3: Calculating Logistic Regression"
		echo "Command: " mahout runlogistic --input "$TMP_LOG_REG_IN_FILE" --model "$TMP_LOG_REG_MODEL_FILE" --auc --confusion --quiet 
		echo "output: hdfs://"$HDFS_RESULT_FILE
		echo "-------------------------"

		mahout runlogistic --input "$TMP_LOG_REG_IN_FILE" --model "$TMP_LOG_REG_MODEL_FILE" --auc --confusion --quiet 2> /dev/null | grep -A 3 "AUC =" | hadoop fs -copyFromLocal -f - "$HDFS_RESULT_FILE"

		echo "-------------------------"
		echo "$QUERY_NAME Step 3/4 Part 4: Cleanup tmp files"
		echo "-------------------------"
		rm -f "$TMP_LOG_REG_IN_FILE"
		rm -f "$TMP_LOG_REG_MODEL_FILE"
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 4 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 4/4: Clean up"
		echo "========================="
	
		"$BINARY" $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -f "${QUERY_DIR}/cleanup.sql"
		hadoop fs -rm -r "$TEMP_DIR"
	fi
}

query_run_clean_method () {
	"$BINARY" $HIVE_PARAMS -i "$COMBINED_PARAMS_FILE" -e "DROP TABLE IF EXISTS $TEMP_TABLE; DROP TABLE IF EXISTS $RESULT_TABLE;"
}
