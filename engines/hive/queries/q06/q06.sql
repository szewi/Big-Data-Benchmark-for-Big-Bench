--Find customers who spend more money via web than in
--stores for a given year. Report customers first name, last name, their country of
--origin and identify if they are preferred customer.

-- Resources

-- Part 1 helper table(s) --------------------------------------------------------------

-- Union customer store_sales and customer web_sales
-- !echo Drop ${hiveconf:TEMP_TABLE};
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};

-- !echo create ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE} (
  customer_id           STRING,
  customer_first_name   STRING,
  customer_last_name    STRING,
  c_preferred_cust_flag STRING,
  c_birth_country       STRING,
  c_login               STRING,
  c_email_address       STRING,
  dyear                 INT,
  year_total            DOUBLE,
  sale_type             STRING
)
;

-- !echo INSERT INTO TABLE ${hiveconf:TEMP_TABLE} -> customer store_sales;
INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
-- customer store_sales
SELECT
  c_customer_id AS customer_id,
  c_first_name  AS customer_first_name,
  c_last_name   AS customer_last_name,
  c_preferred_cust_flag,
  c_birth_country,
  c_login,
  c_email_address,
  sv.d_year     AS dyear,
  sv.year_total AS year_total,
  's'           AS sale_type
FROM (
  SELECT
    ss.ss_customer_sk AS customer_sk,
    dt.d_year AS d_year,
    SUM(
      ((ss_ext_list_price - ss_ext_wholesale_cost
      - ss_ext_discount_amt)
      + ss_ext_sales_price) / 2
    ) AS year_total
  FROM store_sales ss
  INNER JOIN (
    SELECT d_year, d_date_sk
    FROM date_dim
    WHERE d_year >= ${hiveconf:q06_YEAR}
    AND   d_year <= ${hiveconf:q06_YEAR} + 1
  ) dt ON (ss.ss_sold_date_sk = dt.d_date_sk)
  GROUP BY ss.ss_customer_sk, dt.d_year
) sv
INNER JOIN customer c  ON c.c_customer_sk = sv.customer_sk
;


-- !echo INSERT INTO TABLE ${hiveconf:TEMP_TABLE} -> customer web_sales;
INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
-- customer web_sales
SELECT
  c_customer_id  AS customer_id,
  c_first_name   AS customer_first_name,
  c_last_name    AS customer_last_name,
  c_preferred_cust_flag,
  c_birth_country,
  c_login,
  c_email_address,
  sv2.d_year     AS dyear,
  sv2.year_total AS year_total,
  'c' AS sale_type
FROM (
  SELECT
    ws.ws_bill_customer_sk AS customer_sk,
    dt.d_year AS d_year,
    SUM(
      ((ws_ext_list_price - ws_ext_wholesale_cost
      - ws_ext_discount_amt)
      + ws_ext_sales_price) / 2
    ) AS year_total
  FROM web_sales ws
  INNER JOIN (
    SELECT d_date_sk, d_year
    FROM date_dim
    WHERE d_year >= ${hiveconf:q06_YEAR}
    AND   d_year <= ${hiveconf:q06_YEAR} + 1
  ) dt ON (ws.ws_sold_date_sk = dt.d_date_sk)
  GROUP BY ws.ws_bill_customer_sk, dt.d_year
) sv2
INNER JOIN customer c ON c.c_customer_sk = sv2.customer_sk
;


--Part2: self-joins
----hive 0.8.1 does not support self-joins
----if you have this version: create 4 different tables with same values to carry out the task
--DROP VIEW IF EXISTS q06_t_s_firstyear;
--DROP VIEW IF EXISTS q06_t_s_secyear;
--DROP VIEW IF EXISTS q06_t_c_firstyear;
--DROP VIEW IF EXISTS q06_t_c_secyear;

--CREATE VIEW IF NOT EXISTS q06_t_s_firstyear AS SELECT * FROM ${hiveconf:TEMP_TABLE};
--CREATE VIEW IF NOT EXISTS q06_t_s_secyear   AS SELECT * FROM ${hiveconf:TEMP_TABLE};
--CREATE VIEW IF NOT EXISTS q06_t_c_firstyear AS SELECT * FROM ${hiveconf:TEMP_TABLE};
--CREATE VIEW IF NOT EXISTS q06_t_c_secyear   AS SELECT * FROM ${hiveconf:TEMP_TABLE};

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  customer_id           STRING,
  customer_first_name   STRING,
  customer_last_name    STRING,
  c_preferred_cust_flag STRING,
  c_birth_country       STRING,
  c_login               STRING,
  cnt1                  DOUBLE,
  cnt2                  DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT
  ts_s.customer_id,
  ts_s.customer_first_name,
  ts_s.customer_last_name,
  ts_s.c_preferred_cust_flag,
  ts_s.c_birth_country,
  ts_s.c_login,
  CASE WHEN tc_f.year_total > 0
    THEN tc_s.year_total / tc_f.year_total
    ELSE null END,
CASE WHEN ts_f.year_total > 0
    THEN ts_s.year_total / ts_f.year_total
    ELSE null END

--FROM q06_t_s_firstyear ts_f
--  INNER JOIN q06_t_s_secyear ts_s ON ts_f.customer_id = ts_s.customer_id
--  INNER JOIN q06_t_c_secyear tc_s ON ts_f.customer_id = tc_s.customer_id
--  INNER JOIN q06_t_c_firstyear tc_f ON ts_f.customer_id = tc_f.customer_id
FROM ${hiveconf:TEMP_TABLE} ts_f
INNER JOIN ${hiveconf:TEMP_TABLE} ts_s ON ts_f.customer_id = ts_s.customer_id
INNER JOIN ${hiveconf:TEMP_TABLE} tc_s ON ts_f.customer_id = tc_s.customer_id
INNER JOIN ${hiveconf:TEMP_TABLE} tc_f ON ts_f.customer_id = tc_f.customer_id

WHERE ts_f.sale_type  = 's'
AND tc_f.sale_type  = 'c'
AND ts_s.sale_type  = 's'
AND tc_s.sale_type  = 'c'
AND ts_f.dyear      = ${hiveconf:q06_YEAR}
AND ts_s.dyear      = ${hiveconf:q06_YEAR} + 1
AND tc_f.dyear      = ${hiveconf:q06_YEAR}
AND tc_s.dyear      = ${hiveconf:q06_YEAR} + 1
AND ts_f.year_total > 0
AND tc_f.year_total > 0

CLUSTER BY
  ts_s.customer_id,
  ts_s.customer_first_name,
  ts_s.customer_last_name,
  ts_s.c_preferred_cust_flag,
  ts_s.c_birth_country,
  ts_s.c_login
LIMIT ${hiveconf:q06_LIMIT};


---Cleanup-------------------------------------------------------------------

--DROP VIEW IF EXISTS q06_t_s_firstyear;
--DROP VIEW IF EXISTS q06_t_s_secyear;
--DROP VIEW IF EXISTS q06_t_c_firstyear;
--DROP VIEW IF EXISTS q06_t_c_secyear;
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
