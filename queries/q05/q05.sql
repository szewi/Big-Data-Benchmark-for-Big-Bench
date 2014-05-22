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

-- Result file configuration
set QUERY_NUM=q05;


--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS q05_ctable;
CREATE TABLE q05_ctable 
	--( c_customer_sk 	STRING
	--, college_education 	STRING
	--, male 		STRING
	--, label 		STRING ) 
ROW FORMAT DELIMITED 
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '${hiveconf:MH_TMP_DIR}'
AS
-- the real query part


	SELECT 	q05_tmp_Cust.c_customer_sk , 
		q05_tmp_Cust.college_education , 
		q05_tmp_Cust.male, 
		CASE WHEN q05_tmp_Cust.clicks_in_category > 2 THEN 1 ELSE 0 END AS label
	FROM (
		SELECT 	q05_tmp_cust_clicks.c_customer_sk 	AS c_customer_sk, 
			q05_tmp_cust_clicks.college_education 	AS college_education, 
			q05_tmp_cust_clicks.male 		AS male, 
			SUM(CASE WHEN q05_tmp_cust_clicks.i_category ='Books' 
				THEN 1 ELSE 0 END) 		AS clicks_in_category
		FROM ( 
			SELECT 	ct.c_customer_sk 		AS c_customer_sk, 
				CASE WHEN cdt.cd_education_status IN ('Advanced Degree', 'College', '4 yr Degree', '2 yr Degree') 
					THEN 1 
					ELSE 0 END 		AS college_education, 
				CASE WHEN cdt.cd_gender = 'M' 
					THEN 1 
					ELSE 0 END 		AS male,
				 it.i_category 			AS i_category
			FROM customer ct
			INNER JOIN customer_demographics cdt  ON ct.c_current_cdemo_sk  = cdt.cd_demo_sk
			INNER JOIN web_clickstreams 	 wcst ON wcst.wcs_user_sk 	= ct.c_customer_sk  
			INNER JOIN item 		 it   ON wcst.wcs_item_sk 	= it.i_item_sk

		) q05_tmp_cust_clicks
		GROUP BY 	q05_tmp_cust_clicks.c_customer_sk, 
				q05_tmp_cust_clicks.college_education, 
				q05_tmp_cust_clicks.male
	) q05_tmp_Cust
;


