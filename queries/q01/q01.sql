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
set QUERY_NUM=q01;
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
--Find the most frequent ones
SELECT	pid1, pid2, COUNT (*) AS cnt
FROM (
	--Make items basket
	FROM (
		-- Joining two tables
		FROM (
			SELECT s.ss_ticket_number AS oid , s.ss_item_sk AS pid
			FROM store_sales s
			INNER JOIN item i ON s.ss_item_sk = i.i_item_sk
			WHERE i.i_category_id in (1 ,2 ,3) and s.ss_store_sk in (10 , 20, 33, 40, 50)
		) q01_temp_join
		MAP q01_temp_join.oid, q01_temp_join.pid
		USING 'cat'
		AS oid, pid 
		CLUSTER BY oid
	) q01_map_output
	REDUCE q01_map_output.oid, q01_map_output.pid
	USING 'java -cp bigbenchqueriesmr.jar:hive-contrib.jar de.bankmark.bigbench.queries.q01.Red'
	AS (pid1 BIGINT, pid2 BIGINT)
) q01_temp_basket
GROUP BY pid1, pid2
HAVING COUNT (pid1) > 49
ORDER BY pid1 ,cnt ,pid2;
