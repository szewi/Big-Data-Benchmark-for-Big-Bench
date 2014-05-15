#!/usr/bin/env bash

#find $BIG_BENCH_HOME/logs/q??.log -type f -print | xargs grep -A 10 "time ===="
grep -A 10 "time ====" $BIG_BENCH_HOME/logs/q??.log
