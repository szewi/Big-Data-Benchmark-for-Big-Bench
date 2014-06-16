#!/usr/bin/env bash
grep "$BIG_BENCH_HDFS_MOUNT_POINT" /etc/mtab > /dev/null 2>&1 && sudo umount "$BIG_BENCH_HDFS_MOUNT_POINT"
sudo [ ! -d "$BIG_BENCH_HDFS_MOUNT_POINT" ] && mkdir -p "$BIG_BENCH_HDFS_MOUNT_POINT"
sudo hadoop-fuse-dfs dfs://${BIG_BENCH_HDFS_NAMENODE} "$BIG_BENCH_HDFS_MOUNT_POINT"
