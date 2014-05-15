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
add file ${env:BIG_BENCH_HIVE_LIBS}/hive-contrib.jar;
add file ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;

-- Result file configuration
set QUERY_NUM=q02;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};

--Result -------------------------------------------------------------------------
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:resultFile}' 
AS
-- the real query part
SELECT  pid1, pid2, COUNT (*) AS cnt
FROM (
	--Make items basket
	FROM (
		-- Select predicate
		FROM (
		        SELECT wcs_user_sk AS cid , wcs_item_sk AS pid
		        FROM web_clickstreams
		        WHERE wcs_item_sk IS NOT NULL AND wcs_user_sk IS NOT NULL

		) q02_temp
		MAP q02_temp.cid, q02_temp.pid
		USING 'cat'
		AS cid, pid
		CLUSTER BY cid
	) q02_map_output
	REDUCE q02_map_output.cid, q02_map_output.pid
	USING 'java -cp bigbenchqueriesmr.jar:hive-contrib.jar de.bankmark.bigbench.queries.q02.Red'
	AS (pid1 BIGINT, pid2 BIGINT)
) q02_temp_basket
WHERE pid1 = 1416
GROUP BY pid1, pid2
ORDER BY pid1 ,cnt ,pid2
LIMIT 30;
