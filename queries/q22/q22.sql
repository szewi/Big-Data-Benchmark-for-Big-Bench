--For all items whose price was changed on a given date,
--compute the percentage change in inventory between the 30-day period BEFORE
--the price change and the 30-day period AFTER the change. Group this
--information by warehouse.

-- Resources

--Result --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  w_warehouse_name STRING,
  i_item_id        STRING,
  inv_before       BIGINT,
  inv_after        BIGINT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT *
FROM (
  SELECT
    w_warehouse_name,
    i_item_id,
    SUM(
      CASE WHEN datediff(d_date, '${hiveconf:q22_date}') < 0
      THEN inv_quantity_on_hand
      ELSE 0 END
    ) AS inv_before,
    SUM(
      CASE WHEN datediff(d_date, '${hiveconf:q22_date}') >= 0
      THEN inv_quantity_on_hand
      ELSE 0 END
    ) AS inv_after
  FROM (
    SELECT *
    FROM inventory inv
    JOIN (
      SELECT
        i_item_id,
        i_item_sk
      FROM item
      WHERE i_current_price > ${hiveconf:q22_i_current_price_min}
      AND i_current_price < ${hiveconf:q22_i_current_price_max}
    ) items
    ON inv.inv_item_sk = items.i_item_sk
    JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
    JOIN date_dim d ON inv.inv_date_sk = d.d_date_sk
    WHERE datediff(d_date, '${hiveconf:q22_date}') >= -30
    AND datediff(d_date, '${hiveconf:q22_date}') <= 30
  ) q22_coalition_22
  GROUP BY w_warehouse_name, i_item_id
) name
WHERE inv_before > 0
AND inv_after / inv_before >= 2.0 / 3.0
AND inv_after / inv_before <= 3.0 / 2.0
CLUSTER BY
  w_warehouse_name,
  i_item_id
;


---- cleanup ----------------
