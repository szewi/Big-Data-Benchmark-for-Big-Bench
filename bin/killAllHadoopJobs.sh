#!/usr/bin/env bash

for job in `hadoop job -list 2> /dev/null | grep 'job_' | awk '{print $1}'`; do 
  echo "killing: $job"
  hadoop job -kill $job & 
done
wait
echo "They are all dead, Jim"
