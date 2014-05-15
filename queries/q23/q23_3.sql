-- Global hive options (see: Big-Bench/setEnvVars)
set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources
set QUERY_NUM=q23;
set resultTableName2=${hiveconf:QUERY_NUM}result2;
set resultFile2=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName2};

--- RESULT PART 2--------------------------------------
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName2};
CREATE TABLE ${hiveconf:resultTableName2}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:resultFile2}' 
AS
-- Beginn: the real query part
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
FROM q23_tmp_inv inv1 
JOIN q23_tmp_inv inv2 	ON inv1.i_item_sk = inv2.i_item_sk
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

