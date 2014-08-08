--This query contains multiple, related iterations: Iteration 1: Calculate the coeficient of variation 
--and mean of every item and warehouse of two consecutive months Iteration 2: Find items that had a coeficient
--of variation in the first months of 1.5 or larger

-- Resources

--- RESULT PART 2--------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE2};
CREATE TABLE ${hiveconf:RESULT_TABLE2} (
  inv1_w_warehouse_sk BIGINT,
  inv1_i_item_sk      BIGINT,
  inv1_d_moy          INT,
  inv1_mean           DOUBLE,
  inv1_cov            DOUBLE,
  inv2_w_warehouse_sk BIGINT,
  inv2_i_item_sk      BIGINT,
  inv2_d_moy          INT,
  inv2_mean           DOUBLE,
  inv2_cov            DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR2}';

-- Begin: the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE2}
SELECT
  inv1.w_warehouse_sk AS inv1_w_warehouse_sk,
  inv1.i_item_sk AS inv1_i_item_sk,
  inv1.d_moy AS inv1_d_moy,
  inv1.mean AS inv1_mean,
  inv1.cov AS inv1_cov,
  inv2.w_warehouse_sk AS inv2_w_warehouse_sk,
  inv2.i_item_sk AS inv2_i_item_sk,
  inv2.d_moy AS inv2_d_moy,
  inv2.mean AS inv2_mean,
  inv2.cov AS inv2_cov
FROM ${hiveconf:TEMP_TABLE} inv1
JOIN ${hiveconf:TEMP_TABLE} inv2 ON (
  inv1.i_item_sk = inv2.i_item_sk
  AND inv1.w_warehouse_sk = inv2.w_warehouse_sk
  AND inv1.d_moy = ${hiveconf:q23_month} + 1
  AND inv2.d_moy = ${hiveconf:q23_month} + 2
  AND inv1.cov > ${hiveconf:q23_coeficient}
)
CLUSTER BY
  inv1_w_warehouse_sk,
  inv1_i_item_sk,
  inv1_d_moy,
  inv1_mean,
  inv1_cov,
  inv2_d_moy,
  inv2_mean,
  inv2_cov
;
