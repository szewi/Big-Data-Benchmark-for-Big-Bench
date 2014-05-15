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
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q30/mapper_q30.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q30/reducer_q30.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q30/mapper2_q30.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q30/reducer2_q30.py;

set QUERY_NUM=q30;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};


--Result  --------------------------------------------------------------------		
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:resultFile}' 
AS
-- Beginn: the real query part
SELECT ro2.item_sk, ro2.affine_item_sk, ro2.item, ro2.affine_item, ro2.frequency 
FROM (
	FROM (
		FROM (
			FROM (
				FROM web_clickstreams wcs JOIN item i ON wcs.wcs_item_sk = i.i_item_sk
				AND wcs.wcs_user_sk IS NOT NULL
				AND wcs.wcs_item_sk IS NOT NULL
				MAP wcs.wcs_user_sk, wcs.wcs_click_date_sk, i.i_item_sk, i.i_item_id
				USING 'python mapper_q30.py'
				AS key, item_sk, item
				CLUSTER BY key
			) mo
			REDUCE mo.key, mo.item_sk, mo.item
			USING 'python reducer_q30.py'
			AS (item_id, item, affine_item_id, 
			affine_item)
		) ro
		MAP ro.item_id, ro.item, ro.affine_item_id, ro.affine_item
		USING 'python mapper2_q30.py'
		AS combined_key, item_id, item, affine_item_id, affine_item, frequency
		CLUSTER BY combined_key
	) mo2
	REDUCE mo2.combined_key, mo2.item_id, mo2.item, mo2.affine_item_id, mo2.affine_item, mo2.frequency
	USING 'python reducer2_q30.py'
	AS (item_sk, item, affine_item_sk, 
	affine_item, frequency)
) ro2
ORDER BY ro2.frequency;



