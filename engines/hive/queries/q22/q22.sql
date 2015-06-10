--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.

--based on tpc-ds q21
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
--original was ORDER BY w_warehouse_name, i_item_id , but CLUSTER BY is hives cluster scale counter part
CLUSTER BY w_warehouse_name,
           i_item_id
LIMIT 100
;


---- cleanup ----------------
