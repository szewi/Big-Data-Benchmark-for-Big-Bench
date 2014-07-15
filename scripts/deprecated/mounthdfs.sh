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

grep "$BIG_BENCH_HDFS_MOUNT_POINT" /etc/mtab > /dev/null 2>&1 && sudo umount "$BIG_BENCH_HDFS_MOUNT_POINT"
sudo [ ! -d "$BIG_BENCH_HDFS_MOUNT_POINT" ] && mkdir -p "$BIG_BENCH_HDFS_MOUNT_POINT"
sudo hadoop-fuse-dfs dfs://${BIG_BENCH_HDFS_NAMENODE} "$BIG_BENCH_HDFS_MOUNT_POINT"
