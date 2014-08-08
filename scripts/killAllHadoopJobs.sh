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

for job in `hadoop job -list 2> /dev/null | grep 'job_' | awk '{print $1}'`; do 
  echo "killing: $job"
  hadoop job -kill $job & 
done
wait
echo "they are all  dead jim"
