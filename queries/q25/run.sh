#!/usr/bin/env bash

TEMP_RESULT_TABLE="${TABLE_PREFIX}_temp_result"
TEMP_RESULT_DIR="$TEMP_DIR/$TEMP_RESULT_TABLE"

HIVE_PARAMS="$HIVE_PARAMS -hiveconf TEMP_RESULT_TABLE=$TEMP_RESULT_TABLE -hiveconf TEMP_RESULT_DIR=$TEMP_RESULT_DIR"

query_run_main_method () {
	HIVE_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$HIVE_SCRIPT" ]
	then
		echo "SQL file $HIVE_SCRIPT can not be read."
		exit 1
	fi

	#EXECUTION Plan:
	#step 1.  rm/mkdir TEMP_RESULT_DIR	:	hadoop fs rm/mkdir TEMP_RESULT_DIR
	#step 2.  hive q25.sql		:	Run hive querys to extract kmeans input data
	#step 3.  mahout input		:	Generating sparse vectors
	#step 4.  mahout kmeans		:	Calculating k-means"
	#step 5.  mahout dump > hdfs/res:	Converting result and copy result do hdfs query result folder
	#step 6.  hive && hdfs 		:	cleanup.sql && hadoop fs rm MH

	MAHOUT_TEMP_DIR="$TEMP_DIR/mahout_temp"

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 1/6: Prepare temp dir"
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
		echo "$QUERY_NAME Step 2/6: Executing hive queries"
		echo "tmp output: ${TEMP_RESULT_DIR}"
		echo "========================="
		# Write input for k-means into temp table
		runHiveCmd -f "$HIVE_SCRIPT"
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 3 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 3/6: Generating sparse vectors"
		echo "Command "mahout org.apache.mahout.clustering.conversion.InputDriver -i "${TEMP_RESULT_DIR}" -o "${TEMP_RESULT_DIR}/Vec" -v org.apache.mahout.math.RandomAccessSparseVector #-c UTF-8 
		echo "tmp output: ${TEMP_RESULT_DIR}/Vec"
		echo "========================="

		mahout org.apache.mahout.clustering.conversion.InputDriver -i "${TEMP_RESULT_DIR}" -o "${TEMP_RESULT_DIR}/Vec" -v org.apache.mahout.math.RandomAccessSparseVector #-c UTF-8 
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 4 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 4/6: Calculating k-means"
		echo "Command "mahout kmeans -i "$TEMP_RESULT_DIR/Vec" -c "$TEMP_RESULT_DIR/init-clusters" -o "$TEMP_RESULT_DIR/kmeans-clusters" -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
		echo "tmp output: $TEMP_RESULT_DIR/kmeans-clusters"
		echo "========================="

		mahout kmeans --tempDir "$MAHOUT_TEMP_DIR" -i "$TEMP_RESULT_DIR/Vec" -c "$TEMP_RESULT_DIR/init-clusters" -o "$TEMP_RESULT_DIR/kmeans-clusters" -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 5 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 5/6: Converting result and store in hdfs ${RESULT_DIR}/cluster.txt"
		echo "command: mahout clusterdump -i $TEMP_RESULT_DIR/kmeans-clusters/clusters-*-final  -dm org.apache.mahout.common.distance.CosineDistanceMeasure -of TEXT | hadoop fs -copyFromLocal - ${RESULT_DIR}/cluster.txt"
		echo "========================="
	
		mahout clusterdump --tempDir "$MAHOUT_TEMP_DIR" -i "$TEMP_RESULT_DIR"/kmeans-clusters/clusters-*-final  -dm org.apache.mahout.common.distance.CosineDistanceMeasure -of TEXT | hadoop fs -copyFromLocal - "${RESULT_DIR}/cluster.txt"
		#mahout seqdumper --tempDir "$MAHOUT_TEMP_DIR" -i $TEMP_RESULT_DIR/Vec/ -c $TEMP_RESULT_DIR/kmeans-clusters -o $TEMP_RESULT_DIR/results -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 6 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 6/6: Clean up"
		echo "========================="
		runHiveCmd -f "${QUERY_DIR}/cleanup.sql"
		hadoop fs -rm -r "$TEMP_RESULT_DIR"
	fi
}

query_run_clean_method () {
	runHiveCmd -e "DROP TABLE IF EXISTS $TEMP_TABLE; DROP TABLE IF EXISTS $TEMP_RESULT_TABLE; DROP TABLE IF EXISTS $RESULT_TABLE;"
}
