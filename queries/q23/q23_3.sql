-- Global hive options (see: Big-Bench/setEnvVars)
--set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
--set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
--set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
--set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
--set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
--set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};
--set hive.default.fileformat=${env:BIG_BENCH_hive_default_fileformat};
--set hive.optimize.mapjoin.mapreduce=${env:BIG_BENCH_hive_optimize_mapjoin_mapreduce};
--set hive.optimize.bucketmapjoin=${env:BIG_BENCH_hive_optimize_bucketmapjoin};
--set hive.optimize.bucketmapjoin.sortedmerge=${env:BIG_BENCH_hive_optimize_bucketmapjoin_sortedmerge};
--set hive.auto.convert.join=${env:BIG_BENCH_hive_auto_convert_join};
--set hive.auto.convert.sortmerge.join=${env:BIG_BENCH_hive_auto_convert_sortmerge_join};
--set hive.auto.convert.sortmerge.join.noconditionaltask=${env:BIG_BENCH_hive_auto_convert_sortmerge_join_noconditionaltask};
--set hive.optimize.ppd=${env:BIG_BENCH_hive_optimize_ppd};
--set hive.optimize.index.filter=${env:BIG_BENCH_hive_optimize_index_filter};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set hive.optimize.mapjoin.mapreduce;
set hive.mapjoin.smalltable.filesize;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;
set hive.optimize.ppd;
set hive.optimize.index.filter;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources

--- RESULT PART 2--------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE2};
CREATE TABLE ${hiveconf:RESULT_TABLE2}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR2}' 
AS
-- Begin: the real query part
SELECT 	inv1.w_warehouse_sk AS inv1_w_warehouse_sk, 
	inv1.i_item_sk AS inv1_i_item_sk, 
	inv1.d_moy AS inv1_d_moy, 
	inv1.mean AS inv1_mean, 
	inv1.cov AS inv1_cov, 
	inv2.w_warehouse_sk AS inv2_w_warehouse_sk,
       	inv2.i_item_sk AS inv2_i_item_sk,
	inv2.d_moy AS inv2_d_moy,
	inv2.mean AS inv2_mean, 
	inv2.cov AS inv2_cov
FROM ${hiveconf:TEMP_TABLE} inv1 
JOIN ${hiveconf:TEMP_TABLE} inv2 	ON inv1.i_item_sk = inv2.i_item_sk
		AND inv1.w_warehouse_sk =  inv2.w_warehouse_sk
		AND inv1.d_moy= 2
		AND inv2.d_moy= 2 + 1
		AND inv1.cov > 1.5 
ORDER BY inv1_w_warehouse_sk, 
	inv1_i_item_sk,
	inv1_d_moy, 
	inv1_mean, 
	inv1_cov,
	inv2_d_moy, 
	inv2_mean, 
	inv2_cov;
