--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


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
