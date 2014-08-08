--This query contains multiple, related iterations: Iteration 1: Calculate the coeficient of variation 
--and mean of every item and warehouse of two consecutive months Iteration 2: Find items that had a coeficient
--of variation in the first months of 1.5 or larger


DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE VIEW ${hiveconf:TEMP_TABLE} AS
SELECT
  w_warehouse_name,
  w_warehouse_sk,
  i_item_sk,
  d_moy,
  stdev,
  mean,
  CASE mean WHEN 0.0 THEN NULL ELSE stdev/mean END cov
FROM (
  SELECT
    w_warehouse_name,
    w_warehouse_sk,
    i_item_sk,
    d_moy,
    stddev_samp(inv_quantity_on_hand) stdev,
    avg(inv_quantity_on_hand) mean
  FROM inventory inv
  JOIN date_dim d ON (inv.inv_date_sk = d.d_date_sk
  AND d.d_year = ${hiveconf:q23_year} )
  JOIN item i ON inv.inv_item_sk = i.i_item_sk
  JOIN warehouse w ON inv.inv_warehouse_sk = w.w_warehouse_sk
  GROUP BY
    w_warehouse_name,
    w_warehouse_sk,
    i_item_sk,
    d_moy
) q23_tmp_inv_part
WHERE CASE mean WHEN 0.0
  THEN 0.0
  ELSE stdev/mean END > 1.0
;
