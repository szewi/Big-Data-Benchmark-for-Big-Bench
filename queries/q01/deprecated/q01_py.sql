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

ADD FILE q1_mapper.py;
ADD FILE q1_reducer.py;

set outputTableName=q01result;
--Prepare result storage
DROP TABLE IF EXISTS ${hiveconf:outputTableName};

CREATE EXTERNAL TABLE ${hiveconf:outputTableName}(pid1  BIGINT, pid2 BIGINT, count INT) ROW FORMAT
              DELIMITED FIELDS TERMINATED BY ','
              LINES TERMINATED BY '\n'
              STORED AS TEXTFILE
              LOCATION '${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:outputTableName}';


--Find the most frequent ones
INSERT OVERWRITE TABLE ${hiveconf:outputTableName}
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
			WHERE i.i_category_id in (1 ,4 ,6) and s.ss_store_sk in (10 , 20, 33, 40, 50)
		) temp_join
		MAP temp_join.oid, temp_join.pid
		USING 'python q1_mapper.py'
		AS oid, pid 
		CLUSTER BY oid
	) map_output
	REDUCE map_output.oid, map_output.pid
	USING 'python q1_reducer.py'
	AS (pid1 BIGINT, pid2 BIGINT)
) temp_basket
GROUP BY pid1, pid2
HAVING COUNT (pid1) > 49
ORDER BY pid1 ,cnt ,pid2;
