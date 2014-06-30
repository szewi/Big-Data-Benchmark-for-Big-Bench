!echo ============================;
!echo Most important settings;
!echo ============================;

set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set mapred.max.split.size;
set mapred.min.split.size;
set hive.exec.reducers.bytes.per.reducer; 
set hive.exec.reducers.max;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.optimize.ppd;
set hive.optimize.index.filter;
set hive.auto.convert.join.noconditionaltask.size;
set hive.auto.convert.join;
set hive.optimize.mapjoin.mapreduce;
set hive.mapred.local.mem;
set hive.mapjoin.smalltable.filesize; 
set hive.mapjoin.localtask.max.memory.usage;



!echo ============================;
!echo Complete settings;
!echo ============================;

set -v;
