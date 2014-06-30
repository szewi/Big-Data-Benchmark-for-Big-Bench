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
--ADD FILE ${hiveconf:QUERY_DIR}/mapper_q28.py;

--Result 1 Training table for mahout--------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE1};
CREATE TABLE ${hiveconf:TEMP_TABLE1} 
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '\t'
	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE
	--STORED AS SEQUENCEFILE 
	LOCATION '${hiveconf:TEMP_DIR1}'
AS
SELECT  pr_review_sk,
	CASE pr_review_rating
		when 1 then 'NEG'
		when 2 then 'NEG'
		when 3 then 'NEU'
		when 4 then 'POS'
		when 5 then 'POS'
    	END AS pr_rating,
    pr_review_content
FROM product_reviews
WHERE pmod(pr_review_sk, 5) IN (1,2,3)
--limit 10000
;


--Result 2 Testing table for mahout --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE2};
CREATE TABLE ${hiveconf:TEMP_TABLE2}
	ROW FORMAT DELIMITED
	FIELDS TERMINATED BY '\t'
	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE
	--STORED AS SEQUENCEFILE 
	LOCATION '${hiveconf:TEMP_DIR2}'
AS
SELECT  pr_review_sk,
	CASE pr_review_rating
		when 1 then 'NEG'
		when 2 then 'NEG'
		when 3 then 'NEU'
		when 4 then 'POS'
		when 5 then 'POS'
    	END AS pr_rating,
    pr_review_content
FROM product_reviews
WHERE pmod(pr_review_sk, 5) in (0,4)
--limit 10000
;
