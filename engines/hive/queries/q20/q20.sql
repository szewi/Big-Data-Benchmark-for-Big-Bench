--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


--Customer segmentation for return analysis: Customers are separated
--along the following dimensions: return frequency, return order ratio (total num-
--ber of orders partially or fully returned versus the total number of orders),
--return item ratio (total number of items returned versus the number of items
--purchased), return amount ration (total monetary amount of items returned ver-
--sus the amount purchased), return order ratio. Consider the store returns during
--a given year for the computation.

-- Resources

------ create input table for mahout --------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE} (
  cid            INT,
  r_order_ratio  decimal(15,7),
  r_item_ratio   decimal(15,7),
  r_amount_ratio decimal(15,7),
  r_freq         INT
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}';

-- in case you wonder about the structure of the select part
-- hive 0.12 has a bug requiring data-type of x == data-type of y in: "CASE WHEN <condition> THEN x ELSE y"
-- because auto-up-casting will not work!
-- e.g. let x be some table column of type BigInt and y = 1 then:
-- "CASE WHEN <condition> THEN x ELSE 1" will fail because 1 is "int" and not "bigint" but:
-- "CASE WHEN <condition> THEN x ELSE 1L" will work because 1L is of type "bigint"
INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
SELECT
	  cid,
	  100.0 * COUNT(distinct (CASE WHEN r_date IS NOT NULL THEN oid ELSE 0L END)) / COUNT(distinct oid) AS r_order_ratio,
	  SUM(CASE WHEN r_date IS NOT NULL THEN 1 ELSE 0 END) / COUNT(item) * 100 AS r_item_ratio,
	  CASE WHEN SUM(s_amount)=0.0 THEN 0.0 ELSE (SUM(CASE WHEN r_date IS NOT NULL THEN r_amount ELSE 0.0 END) / SUM(s_amount) * 100) END AS r_amount_ratio,
	  COUNT(distinct (CASE WHEN r_date IS NOT NULL THEN r_date ELSE 0L END)) AS r_freq
FROM 
(
  SELECT
		r.sr_returned_date_sk AS r_date,
		s.ss_item_sk AS item,
		s.ss_ticket_number AS oid,
		s.ss_net_paid AS s_amount,
		CASE WHEN r.sr_return_amt IS NULL THEN 0.0 ELSE r.sr_return_amt END AS r_amount,
		(CASE WHEN s.ss_customer_sk  IS NULL THEN r.sr_customer_sk ELSE s.ss_customer_sk END) AS cid

  FROM store_sales s
  LEFT OUTER JOIN store_returns r ON (
			r.sr_item_sk = s.ss_item_sk
			AND r.sr_ticket_number = s.ss_ticket_number
			AND s.ss_sold_date_sk IS NOT NULL
		  )
   WHERE (r.sr_customer_sk IS NOT NULL  OR  s.ss_customer_sk IS NOT NULL)
) q20_sales_returns

GROUP BY cid
HAVING r_freq > 1
Cluster by cid, r_freq ,r_order_ratio, r_item_ratio, r_amount_ratio
--(below) error in data: returns NULL
;

------- Cleanup --------------------------------------
