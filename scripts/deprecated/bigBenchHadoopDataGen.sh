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

# extract options and their arguments into variables.
echo "USAGE: -mapTasks <number> -sf <number>  (SF: scalingFactor 1==1GB, 10==10GB, .. , 1000=1TB, etc. Location of generated data can be set in /setEnvVars configuration file  with the BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR property)"
    case "$1" in
        -mapTasks)
            case "$2" in
                "") echo "Error parsing args. Expected a number after -mapTasks" ; exit 1 ;;
        -*) echo "Error parsing args. Expected a number after -mapTasks" ; exit 1 ;;
                *) HadoopClusterExecOptions="-mapTasks $2" ; shift 2 ;;
            esac ;;
        *) echo "Error parsing args. Expected '-mapTasks <number>' as first argument " ; exit 1 ;;
   
    esac

echo "PDGFOptions: $@"
echo "HadoopClusterExecOptions: $HadoopClusterExecOptions"

PDGF_ARCHIVE_NAME="pdgfEnvironment.tar"
PDGF_DISTRIBUTED_NODE_DIR="$PDGF_ARCHIVE_NAME/data-generator/"
PDGF_ARCHIVE_PATH="$BIG_BENCH_HOME/$PDGF_ARCHIVE_NAME"

if grep -q "IS_EULA_ACCEPTED=true" "$BIG_BENCH_DATA_GENERATOR_DIR/Constants.properties"; then
  echo "EULA is accepted"
else
  echo "==============================================="
  echo "data generator EULA"
  echo "==============================================="
  echo "This is your first run of the data generation tool. Please accept the EULA."
  java -jar "$BIG_BENCH_DATA_GENERATOR_DIR/pdgf.jar" -ns -c
  if grep -q "IS_EULA_ACCEPTED=true" "$BIG_BENCH_DATA_GENERATOR_DIR/Constants.properties"; then
    echo "OK"
  else
    echo "ERROR! data generation tool EULA is not accepted. Cannot procced"
    exit -1 
  fi
fi

# delete any previously generated data
echo "==============================================="
echo "Deleting any previously generated data."
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
echo "make hdfs benchmark refresh data dir: "${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}
echo "==============================================="
hadoop fs -mkdir -p "${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}"
rc=$?
if [[ $rc != 0 ]] ; then
    echo "Error creating hdfs dir: ${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}"
    exit $rc
fi

hadoop fs -chmod -R 777 "${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}"
rc=$?
if [[ $rc != 0 ]] ; then
    echo "Error setting permission for hdfs dir: ${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}"
    exit $rc
fi

hadoop fs -ls "${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}"
echo "OK"

echo "==============================================="
echo "Creating data generator archive to upload to DistCache"
echo "==============================================="
rm -f "$PDGF_ARCHIVE_PATH"
tar -C "$BIG_BENCH_HOME" -caf "$PDGF_ARCHIVE_PATH" data-generator/
rc=$?
if [[ $rc != 0 ]] ; then
    echo "Creating data generator archive failed"
    exit $rc
fi
echo "OK"

echo "==============================================="
echo "Starting distributed hadoop data generation job with: $HadoopClusterExecOptions"
echo "Temporary result data in hdfs: ${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR} (you can change the data generation target folder in  the /setEnvVars configuration file with the BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR property)"
echo "logs: ${BIG_BENCH_DATAGEN_STAGE_LOG}"
echo "==============================================="
HADOOP_CP=`hadoop classpath`
echo "HADOOP CLASSPATH: "$HADOOP_CP

PDGF_CLUSTER_CONF="-Dpdgf.log.folder=/tmp/pdgfLog/HadoopClusterExec.taskNumber -Dcore-site.xml=${BIG_BENCH_DATAGEN_CORE_SITE} -Dhdfs-site.xml=${BIG_BENCH_DATAGEN_HDFS_SITE} -Djava.library.path=${BIG_BENCH_HADOOP_LIBS_NATIVE} -DFileChannelProvider=pdgf.util.caching.fileWriter.HDFSChannelProvider -Ddfs.replication.override=${BIG_BENCH_DATAGEN_DFS_REPLICATION} "
echo "PDGF_CLUSTER_CONF: $PDGF_CLUSTER_CONF"

echo "create $BIG_BENCH_LOGS_DIR folder"
mkdir -p "$BIG_BENCH_LOGS_DIR"

echo hadoop jar "${BIG_BENCH_TOOLS_DIR}/HadoopClusterExec.jar" ${BIGBENCH_DATAGEN_HADOOP_EXEC_DEBUG} -archives2 "${PDGF_ARCHIVE_PATH}" -execCWD "${PDGF_DISTRIBUTED_NODE_DIR}" ${HadoopClusterExecOptions} -exec ${BIG_BENCH_DATAGEN_HADOOP_JVM_ENV} -cp "${HADOOP_CP}:pdgf.jar" ${PDGF_CLUSTER_CONF} pdgf.Controller -nc HadoopClusterExec.tasks  -nn HadoopClusterExec.taskNumber -ns -c -sp REFRESH_PHASE 0 -o "'${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}/'+table.getName()+'/'" ${BIGBENCH_DATAGEN_HADOOP_OPTIONS} -s ${BIGBENCH_TABLES} "$@" 

time (hadoop jar "${BIG_BENCH_TOOLS_DIR}/HadoopClusterExec.jar" ${BIGBENCH_DATAGEN_HADOOP_EXEC_DEBUG} -archives2 "${PDGF_ARCHIVE_PATH}" -execCWD "${PDGF_DISTRIBUTED_NODE_DIR}" ${HadoopClusterExecOptions} -exec ${BIG_BENCH_DATAGEN_HADOOP_JVM_ENV} -cp "${HADOOP_CP}:pdgf.jar" ${PDGF_CLUSTER_CONF} pdgf.Controller -nc HadoopClusterExec.tasks  -nn HadoopClusterExec.taskNumber -ns -c -sp REFRESH_PHASE 0 -o "'${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}/'+table.getName()+'/'" ${BIGBENCH_DATAGEN_HADOOP_OPTIONS} -s ${BIGBENCH_TABLES} "$@"  ; echo  "======= Generating data time =========") > >(tee -a "$BIG_BENCH_DATAGEN_STAGE_LOG") 2>&1 
rc=$?
if [[ $rc != 0 ]] ; then
    echo "Error executing data generation job"
    exit $rc
fi

echo hadoop jar "${BIG_BENCH_TOOLS_DIR}/HadoopClusterExec.jar" ${BIGBENCH_DATAGEN_HADOOP_EXEC_DEBUG} -archives2 "${PDGF_ARCHIVE_PATH}" -execCWD "${PDGF_DISTRIBUTED_NODE_DIR}" ${HadoopClusterExecOptions} -exec ${BIG_BENCH_DATAGEN_HADOOP_JVM_ENV} -cp "${HADOOP_CP}:pdgf.jar" ${PDGF_CLUSTER_CONF} pdgf.Controller -nc HadoopClusterExec.tasks  -nn HadoopClusterExec.taskNumber -ns -c -sp REFRESH_PHASE 1 -o "'${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}/'+table.getName()+'/'" ${BIGBENCH_DATAGEN_HADOOP_OPTIONS} -s ${BIGBENCH_TABLES} "$@" 

time (hadoop jar "${BIG_BENCH_TOOLS_DIR}/HadoopClusterExec.jar" ${BIGBENCH_DATAGEN_HADOOP_EXEC_DEBUG} -archives2 "${PDGF_ARCHIVE_PATH}" -execCWD "${PDGF_DISTRIBUTED_NODE_DIR}" ${HadoopClusterExecOptions} -exec ${BIG_BENCH_DATAGEN_HADOOP_JVM_ENV} -cp "${HADOOP_CP}:pdgf.jar" ${PDGF_CLUSTER_CONF} pdgf.Controller -nc HadoopClusterExec.tasks  -nn HadoopClusterExec.taskNumber -ns -c -sp REFRESH_PHASE 1 -o "'${BIG_BENCH_HDFS_ABSOLUTE_REFRESH_DATA_DIR}/'+table.getName()+'/'" ${BIGBENCH_DATAGEN_HADOOP_OPTIONS} -s ${BIGBENCH_TABLES} "$@"  ; echo  "======= Generating data time =========") > >(tee -a "$BIG_BENCH_DATAGEN_STAGE_LOG") 2>&1 
rc=$?
if [[ $rc != 0 ]] ; then
    echo "Error executing data generation job"
    exit $rc
fi

##cleanup
rm -f ${PDGF_ARCHIVE_PATH}

echo "==============================================="
echo "Hadoop data generation job finished. "
echo "logs: ${BIG_BENCH_DATAGEN_STAGE_LOG}"
echo "View generated files: hadoop fs -ls ${BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR}"
echo "You may start loading the generated tables into HIVE using the script:"
echo " bigBenchPopulateHive.sh"
echo "==============================================="
