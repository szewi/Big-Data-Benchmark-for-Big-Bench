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

-- Result file configuration
set QUERY_NUM=q05;
set resultTableName=ctable;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};


--Result  --------------------------------------------------------------------		
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName} 
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' LINES TERMINATED BY '\n'
STORED AS TEXTFILE  LOCATION '${hiveconf:MH_DIR}'
AS	
-- the real query part
	SELECT 	q05_tmp_Cust.c_customer_sk , 
		q05_tmp_Cust.college_education , 
		q05_tmp_Cust.male, 
		CASE WHEN q05_tmp_Cust.clicks_in_category > 2 THEN true ELSE false END AS label
	FROM 	(
		SELECT 	q05_tmp_cust_clicks.c_customer_sk AS c_customer_sk, 
			q05_tmp_cust_clicks.college_education AS college_education, 
			q05_tmp_cust_clicks.male AS male, 
			SUM(CASE WHEN q05_tmp_cust_clicks.i_category ='Books' THEN 1 ELSE 0 END) AS clicks_in_category
		FROM 	( 
			SELECT 	ct.c_customer_sk AS c_customer_sk, 
				CASE WHEN cdt.cd_education_status IN ('Advanced Degree', 'College', '4 yr Degree', '2 yr Degree') THEN true ELSE false END AS college_education, 
				CASE WHEN cdt.cd_gender = 'M' THEN true ELSE false END AS male, it.i_category AS i_category
			FROM customer ct
			INNER JOIN customer_demographics cdt ON ct.c_current_cdemo_sk = cdt.cd_demo_sk
			INNER JOIN web_clickstreams wcst ON wcst.wcs_user_sk = ct.c_customer_sk  
			INNER JOIN item it ON wcst.wcs_item_sk = it.i_item_sk
			) q05_tmp_cust_clicks
		GROUP BY 	q05_tmp_cust_clicks.c_customer_sk, 
				q05_tmp_cust_clicks.college_education, 
				q05_tmp_cust_clicks.male
		) q05_tmp_Cust
;

