--Get all items that were sold in stores in a given month
--and year and which were returned in the next six months and re-purchased by
--the returning customer afterwards through the web sales channel in the following
--three years. For those these items, compute the total quantity sold through the
--store, the quantity returned and the quantity purchased through the web. Group
--this information by item and store.

-- Resources

--TODO Empty result - needs more testing

--Result --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  item_id                STRING,
  item_desc              STRING,
  store_id               STRING,
  store_name             STRING,
  store_sales_quantity   BIGINT,
  store_returns_quantity BIGINT,
  web_sales_quantity     BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT
  i.i_item_id AS item_id,
  i.i_item_desc AS item_desc,
  s.s_store_id AS store_id,
  s.s_store_name AS store_name,
  SUM(ss.ss_quantity) AS store_sales_quantity,
  SUM(sr.sr_return_quantity) AS store_returns_quantity,
  SUM(ws.ws_quantity) AS web_sales_quantity
FROM store_sales ss

JOIN (
  SELECT d_date_sk
  FROM date_dim
  WHERE d_year=${hiveconf:q21_year}
  AND d_moy=${hiveconf:q21_month}
) d1
ON d1.d_date_sk = ss.ss_sold_date_sk

JOIN store_returns sr
ON sr.sr_customer_sk = ss.ss_customer_sk
AND ss.ss_item_sk = sr.sr_item_sk
AND ss.ss_ticket_number = sr.sr_ticket_number

JOIN (
  SELECT d_date_sk
  FROM date_dim
  WHERE d_year = ${hiveconf:q21_year}
  AND d_moy >= ${hiveconf:q21_month}
  AND d_moy <= ${hiveconf:q21_month} + 3
) d2
ON d2.d_date_sk = sr.sr_returned_date_sk

JOIN web_sales ws
ON  sr.sr_item_sk = ws.ws_item_sk
AND sr.sr_customer_sk = ws.ws_bill_customer_sk

JOIN (
  SELECT d_date_sk
  FROM date_dim
  WHERE d_year in (${hiveconf:q21_year} ,${hiveconf:q21_year} + 1 ,${hiveconf:q21_year} + 2)
) d3
ON d3.d_date_sk = ws.ws_sold_date_sk

JOIN item i ON i.i_item_sk = ss.ss_item_sk
JOIN store s ON s.s_store_sk = ss.ss_store_sk
GROUP BY i.i_item_id, i.i_item_desc, s.s_store_id, s.s_store_name
CLUSTER BY item_id, item_desc, store_id, store_name
;
