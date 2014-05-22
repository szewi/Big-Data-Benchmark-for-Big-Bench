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
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q29/mapper_q29.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q29/reducer_q29.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q29/mapper2_q29.py;
ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q29/reducer2_q29.py;

-- Resources
set QUERY_NUM=q29;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};


--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT
DELIMITED FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table}
LOCATION '${hiveconf:resultFile}' 
AS
-- Begin: the real query part
SELECT 	ro2.category_id, 
	ro2.affine_category_id, 
	ro2.category, 
	ro2.affine_category, 
	ro2.frequency 
FROM 
(
	FROM 
	(
		FROM 
		(
			FROM 
			(
				FROM web_sales ws 
				JOIN item i 	ON ws.ws_item_sk = i.i_item_sk
						AND i.i_category_id IS NOT NULL
				MAP 	ws.ws_order_number, 
					i.i_category_id, 
					i.i_category
				--USING 'python mapper_q29.py' no-op mapper TODO: we could replace the map with a SELECT clause
				USING 'cat'
				AS 	ordernumber, 
					category_id, 
					category
				CLUSTER BY ordernumber
			) mo
			REDUCE 	mo.ordernumber, 
				mo.category_id, 
				mo.category
			USING 'python reducer_q29.py'
			AS 	(category_id,
				 category, 
				 affine_category_id, 
				 affine_category )
		) ro
		MAP 	ro.category_id, 
			ro.category, 
			ro.affine_category_id, 
			ro.affine_category
		USING 'python mapper2_q29.py'
		AS 	combined_key, 
			category_id, 
			category, 
			affine_category_id, 
			affine_category, 
			frequency
		CLUSTER BY combined_key
	) mo2
	REDUCE 	mo2.combined_key, 
		mo2.category_id, 
		mo2.category, 
		mo2.affine_category_id, 
		mo2.affine_category, 
		mo2.frequency
	USING 'python reducer2_q29.py'
	AS (	category_id, 
		category, 
		affine_category_id, 
	  	affine_category, 
		frequency)
) ro2
ORDER BY ro2.frequency;













