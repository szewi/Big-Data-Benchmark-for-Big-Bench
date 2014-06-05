#!/usr/bin/env bash

echo "==============================================="
echo "Adding/Updating generated files to HIVE. (drops old tables)"
echo "==============================================="

time ("${BIG_BENCH_HIVE_SCRIPT_DIR}/hiveCreateLoadORC.sh" ; echo  "======= Load data into hive time =========") > >(tee -a "$BIG_BENCH_LOADING_STAGE_LOG") 2>&1 
echo "==========================="

echo "==============================================="
echo "Cluster job finished. Data is located in hdfs: ${BIG_BENCH_HDFS_ABSOLUTE_DATA_DIR}"
echo "View files: hadoop fs -ls ${BIG_BENCH_HDFS_ABSOLUTE_DATA_DIR}"
echo "HIVE load finished. You may start executing the queries"
echo "==============================================="
