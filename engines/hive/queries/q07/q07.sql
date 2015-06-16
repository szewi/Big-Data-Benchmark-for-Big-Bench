--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


--based on tpc-ds q6
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
--original was ORDER BY cnt , but CLUSTER BY is hives cluster scale counter part
CLUSTER BY cnt
LIMIT ${hiveconf:q07_LIMIT}
;
