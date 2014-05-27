#!/usr/bin/env bash


#To debug start with: 
#> run.sh <step> 
#to execute only the specified step

#EXECUTION Plan:
#step 1.  rm/mkdir MH_TMP_DIR	:	hadoop fs rm/mkdir MH_TMP_DIR
#step 2.  hive q05.sql		:	Run hive querys to extract the input data
#step 3.  mahout TrainLogistic	:	Train logistic regression model
#step 4.  mahout calc log_reg 	:	Calculating logistic regression for input data
#step 5.  mahout dump > hdfs/res:	Converting result and copy result do hdfs query result folder
#step 6.  hive && hdfs 		:	cleanup.sql && hadoop fs rm MH


QUERY_NUM="q05"
QUERY_DIR="${BIG_BENCH_QUERIES_DIR}/${QUERY_NUM}"
MH_TMP_DIR=$BIG_BENCH_HDFS_ABSOLUTE_TEMP_DIR/$QUERY_NUM
HIVE_TABLE_NAME=${MH_TMP_DIR}/ctable

resultTableName=${QUERY_NUM}result
HDFS_RESULT_DIR=${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${resultTableName}
HDFS_RESULT_FILE=${HDFS_RESULT_DIR}/logRegResult.txt

if [ $# -lt 1 ] || [ $1 -eq 1 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 1/4: Prepare temp dir"
	echo "========================="
	hadoop fs -rm -r "$MH_TMP_DIR" &
	hadoop fs -rm -r "$HDFS_RESULT_DIR" &
	wait
	hadoop fs -mkdir "$MH_TMP_DIR" &
	hadoop fs -mkdir "$HDFS_RESULT_DIR" &
	wait
fi


if [ $# -lt 1 ] || [ $1 -eq 2 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 2/4: Executing hive queries"
	echo "tmp output: ${HIVE_TABLE_NAME}"
	echo "========================="
	# Write input for k-means into ctable
	hive -hiveconf "MH_TMP_DIR=${HIVE_TABLE_NAME}" -f "${QUERY_DIR}"/${QUERY_NUM}.sql
fi


if [ $# -lt 1 ] || [ $1 -eq 3 ] ; then
	
	echo "========================="
	echo "$QUERY_NUM Step 3/4: log regression"
	echo "========================="

	TMP_LOG_REG_IN_FILE="`mktemp`"
	TMP_LOG_REG_MODEL_FILE="`mktemp`"

	echo "-------------------------"
	echo "$QUERY_NUN Step 3/4 Part 1: Copy hive result to local csv file"
	echo "tmp output: ${TMP_LOG_REG_IN_FILE}"
	echo "-------------------------"

	echo "streaming result from hive ..."
	#write header
	echo '"c_customer_sk","college_education","male","label"' > "${TMP_LOG_REG_IN_FILE}"
	# append hive result
	hadoop fs -cat "${HIVE_TABLE_NAME}"/* >>   "${TMP_LOG_REG_IN_FILE}"
	echo "streaming result from hive ... done"
	echo "sample:"
	echo "size: " `du -bh "${TMP_LOG_REG_IN_FILE}"`
	echo "------"
	head  "${TMP_LOG_REG_IN_FILE}"
	echo "..." 
	echo "-------------------------"

	echo "$QUERY_NUN Step 3/4 Part 2: Train logistic model"
	echo "Command " mahout trainlogistic --input "$TMP_LOG_REG_IN_FILE" --output "$TMP_LOG_REG_MODEL_FILE" --target c_customer_sk --categories 2 --predictors college_education male label --types n n n --passes 20 --features 20 --rate 1 --lambda 0.5
	echo "tmp output: ${TMP_LOG_REG_MODEL_FILE}"
	echo "-------------------------"
	
	mahout trainlogistic --input "$TMP_LOG_REG_IN_FILE" --output "$TMP_LOG_REG_MODEL_FILE" --target c_customer_sk --categories 2 --predictors college_education male label --types n n n  --passes 20 --features 20 --rate 1 --lambda 0.5	



	echo "-------------------------"
	echo "$QUERY_NUN Step 3/4 Part 3: Calculating Logistic Regression"
	echo "Command: " mahout runlogistic --input "$TMP_LOG_REG_IN_FILE" --model "$TMP_LOG_REG_MODEL_FILE" --auc --confusion --quiet 
	echo "output: hdfs://"$HDFS_RESULT_FILE
	echo "-------------------------"

	mahout runlogistic --input "$TMP_LOG_REG_IN_FILE" --model "$TMP_LOG_REG_MODEL_FILE" --auc --confusion --quiet  2>/dev/null | grep -A 3 "AUC =" | hadoop fs -copyFromLocal -f - "$HDFS_RESULT_FILE"


	echo "-------------------------"
	echo "$QUERY_NUN Step 3/4 Part 4: Cleanup tmp files"
	echo "-------------------------"
	rm -f "$TMP_LOG_REG_IN_FILE"
	rm -f "$TMP_LOG_REG_MODEL_FILE"

fi



if [ $# -lt 1 ] || [ $1 -eq 4 ] ; then
	echo "========================="
	echo "$QUERY_NUM Step 4/4: Clean up"
	echo "========================="
	
	hive -f "${QUERY_DIR}"/cleanup.sql
	hadoop fs -rm -r "$MH_TMP_DIR"
fi


echo "======= $QUERY_NUM  result ======="
echo "results in: ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result"
echo "to display : hadoop fs -cat ${BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${QUERY_NUM}result/*"
echo "========================="






	
