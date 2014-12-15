--Perform category affinity analysis for products purchased online together.

-- Resources

ADD FILE ${hiveconf:QUERY_DIR}/reducer_q29.py;
--ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;

-- Resources

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  category_id        STRING,
  affine_category_id STRING,
  category           STRING,
  affine_category    STRING,
  frequency          BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- Begin: the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT
  ro.category_id AS category_id,
  ro.affine_category_id AS affine_category_id,
  ro.category AS category,
  ro.affine_category AS affine_category,
  count(*) as frequency
FROM (
  FROM (
    SELECT
      ws.ws_order_number AS ordernumber,
      i.i_category_id AS category_id,
      i.i_category AS category
    FROM web_sales ws
    JOIN item i ON (ws.ws_item_sk = i.i_item_sk
    AND i.i_category_id IS NOT NULL)
    CLUSTER BY ordernumber
  ) mo
  REDUCE
    mo.ordernumber,
    mo.category_id,
    mo.category
  USING 'python reducer_q29.py'
--  USING '${env:BIG_BENCH_JAVA} ${env:BIG_BENCH_java_child_process_xmx} -cp bigbenchqueriesmr.jar de.bankmark.bigbench.queries.q29.Red'

  AS (
    category_id,
    category,
    affine_category_id,
    affine_category
  )
) ro
GROUP BY ro.category_id, ro.affine_category_id, ro.category, ro.affine_category
CLUSTER BY frequency
;
