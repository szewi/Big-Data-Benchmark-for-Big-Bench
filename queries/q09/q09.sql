-- Global hive options (see: Big-Bench/setEnvVars)
set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
set hive.exec.compreq09_all_sales.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
set hive.exec.compreq09_all_sales.output=${env:BIG_BENCH_hive_exec_compress_output};
set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compreq09_all_sales.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compreq09_all_sales.output;
set mapred.output.compression.codec;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources

-- Result file configuration
set QUERY_NUM=q09;
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
-- the real query part
SELECT SUM(q09_all_sales.ss_quantity) 
  FROM (
	    SELECT * 
	      FROM store_sales ss1 JOIN customer_address ca1 
		ON ss1.ss_addr_sk = ca1.ca_address_sk 
	       AND ca1.ca_country = 'United States' 
	       AND ca1.ca_state IN ('KY', 'GA', 'NM') 
	       AND 0 <= ss1.ss_net_profit 
	       AND ss1.ss_net_profit <= 2000
	     UNION ALL
	    SELECT * 
	      FROM store_sales ss2 JOIN customer_address ca2 
		ON ss2.ss_addr_sk = ca2.ca_address_sk 
	       AND ca2.ca_country = 'United States' 
	       AND ca2.ca_state IN ('MT', 'OR', 'IN') 
	       AND 150 <= ss2.ss_net_profit 
	       AND ss2.ss_net_profit <= 3000
	     UNION ALL
	    SELECT * 
	      FROM store_sales ss3 JOIN customer_address ca3 
		ON ss3.ss_addr_sk = ca3.ca_address_sk 
	       AND ca3.ca_country = 'United States' 
	       AND ca3.ca_state IN ('WI', 'MO', 'WV') 
	       AND 50 <= ss3.ss_net_profit 
	       AND ss3.ss_net_profit <= 25000
        ) q09_all_sales
  JOIN store s ON s.s_store_sk = q09_all_sales.ss_store_sk
  JOIN date_dim dd ON q09_all_sales.ss_sold_date_sk = dd.d_date_sk 
   AND dd.d_year = 1998
  JOIN customer_demographics cd ON cd.cd_demo_sk = q09_all_sales.ss_cdemo_sk 
   AND cd.cd_marital_status = 'M' 
   AND cd.cd_education_status = '4 yr Degree' 
   AND 50.00 <= q09_all_sales.ss_sales_price 
   AND q09_all_sales.ss_sales_price <= 200.00;

