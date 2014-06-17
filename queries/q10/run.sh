#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

logEnvInformation

HIVE_PARAMS="--auxpath $BIG_BENCH_QUERIES_DIR/Resources/opennlp-maxent-3.0.3.jar:$BIG_BENCH_QUERIES_DIR/Resources/opennlp-tools-1.5.3.jar"
${BIG_BENCH_HIVE_SCRIPT_DIR}/execQuery.sh q10 "${HIVE_PARAMS}"
