#!/usr/bin/env bash

query_run_main_method () {
	QUERY_SCRIPT="$QUERY_DIR/$QUERY_NAME.sql"
	if [ ! -r "$QUERY_SCRIPT" ]
	then
		echo "SQL file $QUERY_SCRIPT can not be read."
		exit 1
	fi

	#EXECUTION Plan:
	#step 1.  hive q26.sql		:	Run hive querys to extract kmeans input data
	#step 2.  mahout input		:	Generating sparse vectors
	#step 3.  mahout kmeans		:	Calculating k-means"
	#step 4.  mahout dump > hdfs/res:	Converting result and copy result do hdfs query result folder
	#step 5.  hive && hdfs 		:	cleanup.sql && hadoop fs rm MH

	MAHOUT_TEMP_DIR="$TEMP_DIR/mahout_temp"

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 1 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 1/5: Executing hive queries"
		echo "tmp output: ${TEMP_DIR}"
		echo "========================="
		# Write input for k-means into temp table
		runCmdWithErrorCheck runEngineCmd -f "$QUERY_SCRIPT"
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 2 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 2/5: Generating sparse vectors"
		echo "Command "mahout org.apache.mahout.clustering.conversion.InputDriver -i "${TEMP_DIR}" -o "${TEMP_DIR}/Vec" -v org.apache.mahout.math.RandomAccessSparseVector #-c UTF-8 
		echo "tmp output: ${TEMP_DIR}/Vec"
		echo "========================="

		runCmdWithErrorCheck mahout org.apache.mahout.clustering.conversion.InputDriver -i "${TEMP_DIR}" -o "${TEMP_DIR}/Vec" -v org.apache.mahout.math.RandomAccessSparseVector #-c UTF-8 
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 3 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 3/5: Calculating k-means"
		echo "Command "mahout kmeans -i "$TEMP_DIR/Vec" -c "$TEMP_DIR/init-clusters" -o "$TEMP_DIR/kmeans-clusters" -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
		echo "tmp output: $TEMP_DIR/kmeans-clusters"
		echo "========================="

		runCmdWithErrorCheck mahout kmeans --tempDir "$MAHOUT_TEMP_DIR" -i "$TEMP_DIR/Vec" -c "$TEMP_DIR/init-clusters" -o "$TEMP_DIR/kmeans-clusters" -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 4 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 4/5: Converting result and store in hdfs ${RESULT_DIR}/cluster.txt"
		echo "command: mahout clusterdump -i $TEMP_DIR/kmeans-clusters/clusters-*-final  -dm org.apache.mahout.common.distance.CosineDistanceMeasure -of TEXT | hadoop fs -copyFromLocal - ${RESULT_DIR}/cluster.txt"
		echo "========================="
	
		runCmdWithErrorCheck mahout clusterdump --tempDir "$MAHOUT_TEMP_DIR" -i "$TEMP_DIR"/kmeans-clusters/clusters-*-final  -dm org.apache.mahout.common.distance.CosineDistanceMeasure -of TEXT | hadoop fs -copyFromLocal - "${RESULT_DIR}/cluster.txt"
		#runCmdWithErrorCheck mahout seqdump -i $TEMP_DIR/Vec/ -c $TEMP_DIR/kmeans-clusters -o $TEMP_DIR/results -dm org.apache.mahout.common.distance.CosineDistanceMeasure -x 10 -k 8 -ow -cl
	fi

	if [[ -z "$DEBUG_QUERY_PART" || $DEBUG_QUERY_PART -eq 5 ]] ; then
		echo "========================="
		echo "$QUERY_NAME Step 5/5: Clean up"
		echo "========================="
		runCmdWithErrorCheck runEngineCmd -f "${QUERY_DIR}/cleanup.sql"
		runCmdWithErrorCheck hadoop fs -rm -r -f "$TEMP_DIR"
	fi
}

query_run_clean_method () {
	runCmdWithErrorCheck runEngineCmd -e "DROP TABLE IF EXISTS $TEMP_TABLE; DROP TABLE IF EXISTS $RESULT_TABLE;"
}
