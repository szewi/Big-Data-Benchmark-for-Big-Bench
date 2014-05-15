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
set QUERY_NUM=q13;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};



DROP TABLE IF EXISTS q13_customer_year_total;
CREATE TABLE q13_customer_year_total
(
customer_id            STRING,
customer_first_name    STRING,
customer_last_name     STRING,
year                   INT,
year_total             DOUBLE,
sale_type              STRING
);



--table contains the values of the intersection of customer table and store_sales tables values 
--that meet the necessary requirements and whose year value is either 1999 or 2000
INSERT INTO TABLE q13_customer_year_total
	SELECT	c_customer_id	AS  customer_id,
		c_first_name	AS  customer_first_name,
		c_last_name	AS  customer_last_name,
		d_year		AS  year,
		sum(ss_net_paid)  AS  year_total,
		's'  		AS sale_type
	FROM customer c
	INNER JOIN (
			SELECT ss_customer_sk, ss_net_paid, dd.d_year 
			FROM store_sales ss 
			LEFT OUTER JOIN date_dim dd ON ss.ss_sold_date_sk = dd.d_date_sk
			WHERE dd.d_year = 1999 OR dd.d_year = 2000
		) ss 
		ON c.c_customer_sk = ss.ss_customer_sk
	GROUP BY
	c_customer_id,
	c_first_name,
	c_last_name,
	d_year;


--table contains the values of the intersection of customer table and web_sales tables values that 
--meet the necessary requirements and whose year value is either 1999 or 2000
INSERT INTO TABLE q13_customer_year_total
	SELECT 	c_customer_id  AS  customer_id,
		c_first_name   AS  customer_first_name,
		c_last_name    AS  customer_last_name,
		d_year         AS  year,
		sum(ws_net_paid)  AS  year_total,
		'w'  AS sale_type
	FROM customer c
	INNER JOIN (
			SELECT ws_bill_customer_sk, ws_net_paid, dd.d_year 
			FROM web_sales ws 
			LEFT OUTER JOIN date_dim dd ON ws.ws_sold_date_sk = dd.d_date_sk
			WHERE dd.d_year = 1999 OR dd.d_year = 2000
		) ws 
	ON c.c_customer_sk = ws.ws_bill_customer_sk
	GROUP BY
	c_customer_id,
	c_first_name,
	c_last_name,
	d_year;


---Set up views required for self joins ----------------------------------------------------------------------
DROP VIEW IF EXISTS q13_t_s_firstyear;
DROP VIEW IF EXISTS q13_t_s_secyear;
DROP VIEW IF EXISTS q13_t_w_firstyear;
DROP VIEW IF EXISTS q13_t_w_secyear;

CREATE VIEW IF NOT EXISTS q13_t_s_firstyear AS
SELECT * FROM q13_customer_year_total;

CREATE VIEW IF NOT EXISTS q13_t_s_secyear AS
SELECT * FROM q13_customer_year_total;

CREATE VIEW IF NOT EXISTS q13_t_w_firstyear AS
SELECT * FROM q13_customer_year_total;

CREATE VIEW IF NOT EXISTS q13_t_w_secyear AS
SELECT * FROM q13_customer_year_total;

--Result  --------------------------------------------------------------------		
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
	ROW FORMAT DELIMITED FIELDS TERMINATED BY ','	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE LOCATION '${hiveconf:resultFile}' 
AS
-- the real query part
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
--  TODO if self joins finally work in hive, use this:
--  q13_customer_year_total  ts_f,
--  q13_customer_year_total ts_s ,
--  q13_customer_year_total tw_f ,
--  q13_customer_year_total q13_t_w_secyear

  q13_t_s_firstyear ts_f
  INNER JOIN q13_t_s_secyear   ts_s ON ts_f.customer_id = ts_s.customer_id
  INNER JOIN q13_t_w_firstyear tw_f ON ts_f.customer_id = tw_f.customer_id
  INNER JOIN q13_t_w_secyear   tw_s ON ts_f.customer_id = tw_s.customer_id 	

WHERE ts_s.customer_id =  ts_f.customer_id
  AND ts_f.customer_id = tw_s.customer_id
  AND ts_f.customer_id = tw_f.customer_id
  AND ts_f.sale_type   = 's'
  AND tw_f.sale_type   = 'w'
  AND ts_s.sale_type   = 's'
  AND tw_s.sale_type   = 'w'
  AND ts_f.year        = 1999
  AND ts_s.year        = 1999 + 1
  AND tw_f.year        = 1999
  AND tw_s.year        = 1999 + 1
  AND ts_f.year_total  > 0
  AND tw_f.year_total  > 0

ORDER BY
 ts_s.customer_id,
 ts_s.customer_first_name,
  ts_s.customer_last_name
LIMIT 100;

--cleanup -----------------------------------------------------------
DROP TABLE IF EXISTS q13_customer_year_total;
DROP VIEW IF EXISTS q13_t_s_firstyear;
DROP VIEW IF EXISTS q13_t_s_secyear;
DROP VIEW IF EXISTS q13_t_w_firstyear;
DROP VIEW IF EXISTS q13_t_w_secyear;
