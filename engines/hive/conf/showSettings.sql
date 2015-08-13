-- !echo ============================;
-- !echo Most important settings;
-- !echo ============================;

-- ============================;
-- Print most important properties;
-- ============================;
--exec engine and optimizer
set hive.execution.engine;
set hive.cbo.enable;
set hive.stats.fetch.partition.stats;
set hive.script.operator.truncate.env;
set hive.compute.query.using.stats;
set hive.vectorized.execution.enabled;
set hive.vectorized.execution.reduce.enabled;
set hive.stats.autogather;
--input output
set mapreduce.input.fileinputformat.split.minsize;
set mapreduce.input.fileinputformat.split.maxsize;
set hive.exec.reducers.bytes.per.reducer; 
set hive.exec.reducers.max;
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set hive.exec.compress.output;
set mapred.map.output.compression.codec;
set mapred.output.compression.codec;
set hive.default.fileformat;
--join optimizations
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join.noconditionaltask.size;
set hive.auto.convert.join;
set hive.optimize.mapjoin.mapreduce;
set hive.mapred.local.mem;
set hive.mapjoin.smalltable.filesize; 
set hive.mapjoin.localtask.max.memory.usage;
set hive.optimize.skewjoin;
set hive.optimize.skewjoin.compiletime;
-- filter optimizations (predicate pushdown to storage level)
set hive.optimize.ppd;
set hive.optimize.ppd.storage;
set hive.ppd.recognizetransivity;
set hive.optimize.index.filter;
--other
set hive.optimize.sampling.orderby=true;
set hive.optimize.sampling.orderby.number;
set hive.optimize.sampling.orderby.percent;
set bigbench.hive.optimize.sampling.orderby;
set bigbench.hive.optimize.sampling.orderby.number;
set bigbench.hive.optimize.sampling.orderby.percent;
set hive.groupby.skewindata;
set hive.exec.submit.local.task.via.child;

-- !echo ============================;
-- !echo Complete settings;
-- !echo ============================;

set -v;
