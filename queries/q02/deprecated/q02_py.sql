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

ADD FILE q2_mapper.py;
ADD FILE q2_reducer.py;

set outputTableName=q02result;
--Prepare result storage
DROP TABLE IF EXISTS ${hiveconf:outputTableName};

CREATE EXTERNAL TABLE ${hiveconf:outputTableName}(pid1  BIGINT, pid2 BIGINT, count INT) ROW FORMAT
              DELIMITED FIELDS TERMINATED BY ','
              LINES TERMINATED BY '\n'
              STORED AS TEXTFILE
              LOCATION '${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:outputTableName}';


--Find the most frequent ones
INSERT OVERWRITE TABLE ${hiveconf:outputTableName}
SELECT	pid1, pid2, COUNT (*) AS cnt
FROM (
	--Make items basket
	FROM (
		-- Select predicate
		FROM (
			SELECT wcs_user_sk AS cid , wcs_item_sk AS pid
			FROM web_clickstreams
			WHERE wcs_item_sk IS NOT NULL AND wcs_user_sk IS NOT NULL
			
		) temp
		MAP temp.cid, temp.pid
		USING 'python q2_mapper.py'
		AS cid, pid 
		CLUSTER BY cid
	) map_output
	REDUCE map_output.cid, map_output.pid
	USING 'python q2_reducer.py'
	AS (pid1 BIGINT, pid2 BIGINT)
) temp_basket
WHERE pid1 = 1416
GROUP BY pid1, pid2
ORDER BY pid1 ,cnt ,pid2
LIMIT 30;
