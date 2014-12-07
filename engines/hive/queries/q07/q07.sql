--List all the stores with at least 10 customers who during
--a given month bought products with the price tag at least 20% higher than the
--average price of products in the same category.

-- Resources

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  state STRING,
  cnt   BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT a.ca_state AS state, COUNT(*) AS cnt
FROM customer_address a
JOIN customer c ON a.ca_address_sk = c.c_current_addr_sk
JOIN store_sales s ON c.c_customer_sk = s.ss_customer_sk
JOIN date_dim d ON s.ss_sold_date_sk = d.d_date_sk
JOIN item i ON s.ss_item_sk = i.i_item_sk
JOIN (
  SELECT DISTINCT(d_month_seq) AS d_month_seq
  FROM date_dim 
  WHERE d_year = ${hiveconf:q07_YEAR}
  AND d_moy = ${hiveconf:q07_MONTH}
) q07_month --subquery alias
ON q07_month.d_month_seq = d.d_month_seq
JOIN (
  SELECT
    i_category AS i_category,
    AVG(i_current_price) * ${hiveconf:q07_HIGHER_PRICE_RATIO} AS avg_price
  FROM item
  GROUP BY i_category
) q07_cat_avg_price --subquery alias
ON q07_cat_avg_price.i_category = i.i_category

WHERE i.i_current_price > q07_cat_avg_price.avg_price
GROUP BY a.ca_state
HAVING COUNT(*) >= ${hiveconf:q07_HAVING_COUNT_GE}
CLUSTER BY cnt
LIMIT ${hiveconf:q07_LIMIT}
;
