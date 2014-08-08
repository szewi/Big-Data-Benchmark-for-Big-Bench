#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
  source "$ENV_SETTINGS"
fi

logEnvInformation

if grep -q "IS_EULA_ACCEPTED=true" "$BIG_BENCH_DATA_GENERATOR_DIR/Constants.properties"; then
  echo "EULA is accepted"
else
  echo "==============================================="
  echo "data generator EULA"
  echo "==============================================="
  echo "This is your first run of the data generation tool. Please accept the EULA."
  ${BIG_BENCH_DATAGEN_JVM_ENV} -jar "$BIG_BENCH_DATA_GENERATOR_DIR"/pdgf.jar -ns -c
  if grep -q "IS_EULA_ACCEPTED=true" "$BIG_BENCH_DATA_GENERATOR_DIR/Constants.properties"; then
    echo "OK"
  else
    echo "ERROR! data generation tool EULA is not accepted. Cannot procced"
    exit 1 
  fi
fi

CLUSTER_CONF=" -Dcore-site.xml=${BIG_BENCH_DATAGEN_CORE_SITE} -Dhdfs-site.xml=${BIG_BENCH_DATAGEN_HDFS_SITE} -Djava.library.path=${BIG_BENCH_HADOOP_LIBS_NATIVE} -DFileChannelProvider=pdgf.util.caching.fileWriter.HDFSChannelProvider -Ddfs.replication.override=${BIG_BENCH_DATAGEN_DFS_REPLICATION} "
#echo $CLUSTER_CONF

IPs=(${BIG_BENCH_NODES})
NODE_COUNT=${#IPs[@]}

echo "==============================================="
echo "Deleting any previously generated data, results and logs."
echo "==============================================="
"${BIG_BENCH_BASH_SCRIPT_DIR}/bigBenchCleanData.sh"
echo "OK"
echo "==============================================="
echo "make hdfs benchmark data dir: "${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}
echo "==============================================="
hadoop fs -mkdir -p "${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
rc=$?
if [[ $rc != 0 ]] ; then
    echo "Error creating hdfs dir: ${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
    exit $rc
fi

hadoop fs -chmod -R 777 "${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
rc=$?
if [[ $rc != 0 ]] ; then
    echo "Error setting permission for hdfs dir: ${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
    exit $rc
fi

hadoop fs -ls "${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
echo "OK"

echo "==============================================="
echo "Starting data generation job."
echo "==============================================="
HADOOP_CP=`hadoop classpath`
echo "HADOOP CLASSPATH: $HADOOP_CP"

for (( i = 0; i < ${NODE_COUNT}; i++ ));
do
  echo ssh ${BIG_BENCH_SSH_OPTIONS} ${IPs[$i]} ${BIG_BENCH_DATAGEN_JVM_ENV} -cp "${HADOOP_CP}:${BIG_BENCH_DATA_GENERATOR_DIR}/pdgf.jar" ${CLUSTER_CONF} pdgf.Controller  -nc ${NODE_COUNT} -nn $((i+1)) -ns -c -o "'${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}/'+table.getName()+'/'" -s ${BIGBENCH_TABLES} "$@"
  ssh ${BIG_BENCH_SSH_OPTIONS} ${IPs[$i]} ${BIG_BENCH_DATAGEN_JVM_ENV} -cp "${HADOOP_CP}:${BIG_BENCH_DATA_GENERATOR_DIR}/pdgf.jar" ${CLUSTER_CONF} pdgf.Controller  -nc ${NODE_COUNT} -nn $((i+1)) -ns -c -o "\'${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}/\'+table.getName\(\)+\'/\'" -s ${BIGBENCH_TABLES} "$@" &

done
wait
echo "==============================================="
echo "SSH cluster data generation job finished. "
echo "View generated files: hadoop fs -ls ${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
echo "You may start loading the generated tables into HIVE using the script:"
echo " bigBenchPopulateHive.sh"
echo "==============================================="
