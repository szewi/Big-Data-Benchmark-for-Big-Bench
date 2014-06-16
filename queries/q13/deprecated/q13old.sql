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
set QUERY_NUM=q13;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};




--------------------------------WORKINGS---------------------------------------------------
--------------------------STORE-SALES-INNER-WORKINGS---------------------------------------
  --isolates all store_sales table values that meet the date requirement
  DROP TABLE IF EXISTS store_sales_date;
  CREATE TABLE store_sales_date AS
  SELECT * 
  FROM store_sales ss
  INNER JOIN date_dim dd ON ss.ss_sold_date_sk = dd.d_date_sk;


  --isolates dates whose year value is 1999 or 2000
  DROP TABLE IF EXISTS dates;
  CREATE TABLE dates AS
  SELECT * FROM date_dim d1 WHERE d1.d_year = 1999 OR d1.d_year = 2000;



  --table contains the values of the intersection of customer table and store_sales tables values 
  --that meet the necessary requirements and whose year value is either 1999 or 2000
  DROP TABLE IF EXISTS cust_inter_store_sales_date;
  CREATE TABLE cust_inter_store_sales_date AS
  SELECT * 
      FROM customer c
      INNER JOIN (SELECT * FROM store_sales_date sd LEFT OUTER JOIN dates d ON sd.d_year = d.d_year) ss ON c.c_customer_sk = ss.ss_customer_sk;

----------------------------COMBINE-WEB-SALES-INNER-WORKINGS-------------------------------
  CREATE TABLE p1 AS
  SELECT  c_customer_id  AS  customer_id,
          c_first_name  AS  customer_first_name,
          c_last_name  AS  customer_last_name,
          d_year  AS  year,
          sum(ss_net_paid)  AS  year_total,
          's'  AS sale_type
  FROM cust_inter_store_sales_date
  GROUP BY
      c_customer_id,
      c_first_name,
      c_last_name,
      d_year;

----------------------------WEB-SALES-INNER-WORKINGS---------------------------------------


  --isolates all web_sales table values that meet the date requirement
  DROP TABLE IF EXISTS web_sales_date;
  CREATE TABLE web_sales_date AS
  SELECT * 
  FROM web_sales ws
  INNER JOIN date_dim dd ON ws.ws_sold_date_sk = dd.d_date_sk;



  --isolates dates whose year value is 1999 or 2000 (dates table remains unchanged)
  DROP TABLE IF EXISTS dates;
  CREATE TABLE dates AS
  SELECT * FROM date_dim d1 WHERE d1.d_year = 1999 OR d1.d_year = 2000;



  --table contains the values of the intersection of customer table and web_sales tables values that 
  --meet the necessary requirements and whose year value is either 1999 or 2000
  DROP TABLE IF EXISTS cust_inter_web_sales_date;
  CREATE TABLE cust_inter_web_sales_date AS
  SELECT * 
      FROM customer c
      INNER JOIN 
	(SELECT * FROM web_sales_date wd LEFT OUTER JOIN dates d ON wd.d_year = d.d_year) ws 
	ON c.c_customer_sk = ws.ws_bill_customer_sk;




--------------------------COMBINE-WEB-SALES-INNER-WORKINGS---------------------------------
  CREATE TABLE p2 AS
  SELECT  c_customer_id  AS  customer_id,
          c_first_name  AS  customer_first_name,
          c_last_name  AS  customer_last_name,
          d_year  AS  year,
          sum(ws_net_paid)  AS  year_total,
          'w'  AS sale_type
  FROM cust_inter_web_sales_date
  GROUP BY
      c_customer_id,
      c_first_name,
      c_last_name,
      d_year;
------------------------------------END-WORKINGS-------------------------------------------
------------------------------------------------------------------------------------------- 
  DROP TABLE IF EXISTS q74_customer_year_total_880;

  CREATE TABLE q74_customer_year_total_880
  (
    customer_id            STRING,
    customer_first_name    STRING,
    customer_last_name     STRING,
    year                   INT,
    year_total             DOUBLE,
    sale_type              STRING
  );

  INSERT TABLE q74_customer_year_total_880
  SELECT * FROM p1

  INSERT INTO TABLE q74_customer_year_total_880
  SELECT * FROM p2;

  --only table that remains: q74_customer_year_total_880
  DROP TABLE store_sales_date;
  DROP TABLE web_sales_date;
  DROP TABLE dates;
  DROP TABLE p1;
  DROP TABLE p2;

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--Part2: self-joins------------------------------------------------------------
DROP TABLE IF EXISTS t_s_firstyear;
DROP TABLE IF EXISTS t_s_secyear;
DROP TABLE IF EXISTS t_w_firstyear;
DROP TABLE IF EXISTS t_w_secyear;
---- set-up
CREATE TABLE IF NOT EXISTS t_s_firstyear AS
SELECT * FROM q74_customer_year_total_880;

CREATE TABLE IF NOT EXISTS t_s_secyear AS
SELECT * FROM q74_customer_year_total_880;

CREATE TABLE IF NOT EXISTS t_w_firstyear AS
SELECT * FROM q74_customer_year_total_880;

CREATE TABLE IF NOT EXISTS t_w_secyear AS
SELECT * FROM q74_customer_year_total_880;
-------------------------------------------------------------------------------

--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	

DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table}  LOCATION '${hiveconf:resultFile}' 
AS
-- Begin: the real query part
SELECT
  ts_s.customer_id,
  ts_s.customer_first_name,
  ts_s.customer_last_name,
  CASE WHEN tw_f.year_total > 0
    THEN tw_s.year_total / tw_f.year_total
  ELSE null END,
  CASE WHEN ts_f.year_total > 0
    THEN ts_s.year_total / ts_f.year_total
  ELSE null END

FROM
  t_s_firstyear ts_f
  INNER JOIN t_s_secyear ts_s ON ts_f.customer_id = ts_s.customer_id
  INNER JOIN t_w_firstyear tw_f ON ts_f.customer_id = tw_f.customer_id
  INNER JOIN t_w_secyear tw_s ON ts_f.customer_id = tw_s.customer_id

WHERE
  ts_f.sale_type        = 's'
  AND tw_f.sale_type    = 'w'
  AND ts_s.sale_type    = 's'
  AND tw_s.sale_type    = 'w'
  AND ts_f.year         = 1999
  AND ts_s.year         = 1999 + 1
  AND tw_f.year         = 1999
  AND tw_s.year         = 1999 + 1
  AND ts_f.year_total   > 0
  AND tw_f.year_total   > 0

ORDER BY
  ts_s.customer_id,
  ts_s.customer_first_name,
  ts_s.customer_last_name
LIMIT 100;

-------------------------------------------------------------------------------
--tear-down

DROP TABLE IF EXISTS q74_customer_year_total_880;
DROP TABLE IF EXISTS t_s_firstyear;
DROP TABLE IF EXISTS t_s_secyear;
DROP TABLE IF EXISTS t_w_firstyear;
DROP TABLE IF EXISTS t_w_secyear;

