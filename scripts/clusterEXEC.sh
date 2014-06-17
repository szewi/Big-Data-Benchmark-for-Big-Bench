#!/usr/bin/env bash

ENV_SETTINGS="`dirname $0`/../setEnvVars"
if [ ! -f "$ENV_SETTINGS" ]
then
        echo "Environment setup file $ENV_SETTINGS not found"
        exit 1
else
        source "$ENV_SETTINGS"
fi

IPs=(${BIG_BENCH_NODES})
NODE_COUNT=${#IPs[@]}
SELF=`hostname -s`

for (( i=0; i<${NODE_COUNT}; i++ ));
do
#  if [ $SELF != ${IPs[$i]} ]
#  then
  echo "NODE: ${IPs[$i]}"
  ssh ${BIG_BENCH_SSH_OPTIONS} -t ${IPs[$i]} $@ 
#  fi
done
