!echo ============================;
!echo <settings from hiveSettings.sql>;
!echo ============================;

-- ###########################
-- output and itermediate table settings 
-- ###########################
-- if you cluster has good cpu's but limited network bandwith, this could speed up the exchange of intermediate results (this option should be turund on if you cluster has high 'net wait i/o%'
-- set hive.exec.compress.intermediate=true;
-- set mapred.map.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec;

-- default is to keep the created result tables human readable.
-- set hive.exec.compress.output=false;
-- set mapred.output.compression.codec=org.apache.hadoop.io.compress.DefaultCodec;

-- set hive.default.fileformat=TextFile;

-- ###########################
-- mappers settings 
-- ###########################
-- Number of mappers used by HIVE, based on table sizes.
-- The number of physical files a table consists of is irrelevant for hives metric for estimating number of mappers. (Hive uses HiveCombineInputFormat, joining the files)
-- the following two parameters are most effective in influencing hives estimation of mappers. To low settings may result in to many map tasks, while to high size settings result in to few map tasks and underutilization of the cluster.
-- both extremes are harmful to the performance. For small data set sizes of 1-100GB a good value  for max.split.size may be 134217728 (128MB). As an estimation, take a medium sized table and divide its size by the number of map tasks you need to utilize your cluster.
set mapred.max.split.size=67108864;
set mapred.min.split.size=1;

-- ###########################
-- reducer settings 
-- ###########################
-- Number of reducers used by HIVE
-- hives metric for estimating reducers is mostly controlled by the following settings. Node: Some Query functions like count(*) or Distinct will lead to hive always using only 1 reducer
-- 1GB default
set hive.exec.reducers.bytes.per.reducer=256000000;
-- set hive.exec.reducers.max=99999;

-- ###########################
-- optimizations for joins. 
-- ###########################
-- things like mapjoins are done in memory and require a lot of it
-- README!!!
-- Hive 0.12 bug, hive ignores  'hive.mapred.local.mem' resulting in out of memory errors in map joins!
-- (more exactly: bug in Hadoop 2.2 where hadoop-env.cmd sets the -xmx parameter multiple times, effectively overriding the user set hive.mapred.locla.mem setting. see: https://issues.apache.org/jira/browse/HADOOP-10245
-- There are 3 workarounds: 
-- 1) assign more memory to the local!! Hadoop JVM client (not! mapred.map.memory)-> map-join child vm will inherit the parents jvm settings
-- 2) reduce "hive.smalltable.filesize" to ~1MB (depends on your cluster settings for the local JVM)
-- 3) turn off "hive.auto.convert.join" to prevent hive from converting the join to a mapjoin.
-- MAP join settings:
-- set hive.auto.convert.join=true;
-- set hive.optimize.mapjoin.mapreduce=true;
-- set hive.mapred.local.mem=1024;
--default:25MB, max size of tables considered for local in memory map joion.Beware! ORC files have only little file size but huge in memory data size! a 25MB ORC easily consumes 512MB.. related: https://issues.apache.org/jira/browse/HIVE-2601
set hive.mapjoin.smalltable.filesize=5000000; 
-- set hive.mapjoin.localtask.max.memory.usage=0.90;

-- set hive.auto.convert.sortmerge.join=true;
-- set hive.auto.convert.sortmerge.join.noconditionaltask=true;
-- set hive.auto.convert.join.noconditionaltask.size=10000;
-- set hive.optimize.bucketmapjoin=true;
-- set hive.optimize.bucketmapjoin.sortedmerge=false;
-- set hive.optimize.skewjoin=true; --READ FIRST: https://issues.apache.org/jira/browse/HIVE-5888
-- set hive.optimize.skewjoin.compiletime=true;
-- set hive.groupby.skewindata=true;

-- ###########################
-- Other tuning options
-- ###########################
-- exec.parallel is still considered unstable, but has the potential to increase you utilization by running multiple independent stages of a query in parallel
-- set hive.exec.parallel=true;
-- set hive.exec.parallel.thread.number=8;

-- predicate pushdown for ORC-files (eager filtering of columns)
-- set hive.optimize.ppd=true;
-- set hive.optimize.index.filter=true;




!echo ============================;
!echo Print most important properties;
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
set hive.optimize.skewjoin;
set hive.optimize.skewjoin.compiletime;
set hive.groupby.skewindata;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

--!echo ============================;
--!echo <dump Complete settings>;
--!echo ============================;
--set -v;
--!echo ============================;
--!echo </dump Complete settings>;
--!echo ============================;
!echo ============================;
!echo </settings from hiveSettings.sql>;
!echo ============================;
