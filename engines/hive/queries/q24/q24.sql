--For a given product, measure the effect of competitor's prices on
--products' in-store and online sales. (Compute the cross-price elasticity of demand
--for a given product)

-- Resources

-- compute the price change % for the competitor
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
CREATE VIEW ${hiveconf:TEMP_TABLE1} AS
SELECT
  i_item_sk, (imp_competitor_price - i_current_price)/i_current_price AS price_change,
  imp_start_date, (imp_end_date - imp_start_date) AS no_days
FROM item i
JOIN item_marketprices imp ON i.i_item_sk = imp.imp_item_sk
WHERE i.i_item_sk IN (${hiveconf:q24_i_item_sk_IN})
AND imp.imp_competitor_price < i.i_current_price
;


DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
CREATE VIEW ${hiveconf:TEMP_TABLE2} AS
SELECT
  ws_item_sk,
  SUM(
    CASE WHEN ws_sold_date_sk >= c.imp_start_date
    AND ws_sold_date_sk < c.imp_start_date + c.no_days
    THEN ws_quantity
    ELSE 0 END
  ) AS current_ws,
  SUM(
    CASE WHEN ws_sold_date_sk >= c.imp_start_date - c.no_days
    AND ws_sold_date_sk < c.imp_start_date
    THEN ws_quantity
    ELSE 0 END
  ) AS prev_ws
FROM web_sales ws
JOIN ${hiveconf:TEMP_TABLE1} c ON ws.ws_item_sk = c.i_item_sk
GROUP BY ws_item_sk
;


DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE3};
CREATE VIEW ${hiveconf:TEMP_TABLE3} AS
SELECT
  ss_item_sk,
  SUM(
    CASE WHEN ss_sold_date_sk >= c.imp_start_date
    AND ss_sold_date_sk < c.imp_start_date + c.no_days
    THEN ss_quantity
    ELSE 0 END
  ) AS current_ss,
  SUM(
    CASE WHEN ss_sold_date_sk >= c.imp_start_date - c.no_days
    AND ss_sold_date_sk < c.imp_start_date
    THEN ss_quantity
    ELSE 0 END
  ) AS prev_ss
FROM store_sales ss
JOIN ${hiveconf:TEMP_TABLE1} c ON c.i_item_sk = ss.ss_item_sk
GROUP BY ss_item_sk
;


--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  i_item_sk               BIGINT,
  cross_price_elasticity  DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- Begin: the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT
  i_item_sk,
  (current_ss + current_ws - prev_ss - prev_ws) / ((prev_ss + prev_ws) * price_change) AS cross_price_elasticity
FROM ${hiveconf:TEMP_TABLE1} c
JOIN ${hiveconf:TEMP_TABLE2} ws ON c.i_item_sk = ws.ws_item_sk
JOIN ${hiveconf:TEMP_TABLE3} ss ON c.i_item_sk = ss.ss_item_sk
;

-- clean up -----------------------------------
DROP VIEW ${hiveconf:TEMP_TABLE2};
DROP VIEW ${hiveconf:TEMP_TABLE3};
DROP VIEW ${hiveconf:TEMP_TABLE1};
