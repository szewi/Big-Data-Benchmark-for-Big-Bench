--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.


-- TASK:
-- Build a model using logistic regression: based on existing users online
-- activities (interest in items of different categories) and demographics, for a visitor to an online store, predict the visitors
-- likelihood to be interested in a given item category.
-- input vectors to the machine learning algorithm are:
--  label             STRING, -- number of clicks in specified category "q05_i_category"
--  college_education STRING, -- has college education [0,1]
--  male              STRING, -- isMale [0,1]
--  clicks_in_1       STRING, -- number of clicks in category id 1
--  clicks_in_2       STRING, -- number of clicks in category id 2
--  clicks_in_7       STRING, -- number of clicks in category id 7
--  clicks_in_4       STRING, -- number of clicks in category id 4
--  clicks_in_5       STRING, -- number of clicks in category id 5
--  clicks_in_6       STRING  -- number of clicks in category id 6
-- TODO: updated this description once improved q5 with more features is merged


-- Resources

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE} (
  label             STRING,
  college_education STRING,
  male              STRING,
  clicks_in_1       STRING,
  clicks_in_2       STRING,
  clicks_in_7       STRING,
  clicks_in_4       STRING,
  clicks_in_5       STRING,
  clicks_in_6       STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}';

-- the real query part

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
SELECT
  q05_tmp_Cust.clicks_in_category,
  q05_tmp_Cust.college_education,
  q05_tmp_Cust.male,
  q05_tmp_Cust.clicks_in_1,
  q05_tmp_Cust.clicks_in_2,
  q05_tmp_Cust.clicks_in_7,
  q05_tmp_Cust.clicks_in_4,
  q05_tmp_Cust.clicks_in_5,
  q05_tmp_Cust.clicks_in_6
FROM (
  SELECT
    q05_tmp_cust_clicks.college_education AS college_education,
    q05_tmp_cust_clicks.male AS male,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category = ${hiveconf:q05_i_category}
      THEN 1
      ELSE 0 END
    ) AS clicks_in_category,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category_id = 1
      THEN 1
      ELSE 0 END
    ) AS clicks_in_1,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category_id = 2
      THEN 1
      ELSE 0 END
    ) AS clicks_in_2,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category_id = 7
      THEN 1
      ELSE 0 END
    ) AS clicks_in_7,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category_id = 4
      THEN 1
      ELSE 0 END
    ) AS clicks_in_4,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category_id = 5
      THEN 1
      ELSE 0 END
    ) AS clicks_in_5,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category_id = 6
      THEN 1
      ELSE 0 END
    ) AS clicks_in_6
  FROM (
    SELECT
      ct.c_customer_sk AS c_customer_sk,
      CASE WHEN cdt.cd_education_status IN (${hiveconf:q05_cd_education_status_IN})
        THEN 1 ELSE 0 END AS college_education,
      CASE WHEN cdt.cd_gender = ${hiveconf:q05_cd_gender}
        THEN 1 ELSE 0 END AS male,
      it.i_category AS i_category,
      it.i_category_id AS i_category_id
    FROM customer ct
    INNER JOIN customer_demographics cdt ON ct.c_current_cdemo_sk = cdt.cd_demo_sk
    INNER JOIN web_clickstreams wcst ON (wcst.wcs_user_sk = ct.c_customer_sk
       AND wcst.wcs_user_sk IS NOT NULL)
    INNER JOIN item it ON wcst.wcs_item_sk = it.i_item_sk
  ) q05_tmp_cust_clicks  
    GROUP BY
    q05_tmp_cust_clicks.c_customer_sk,
    q05_tmp_cust_clicks.college_education,
    q05_tmp_cust_clicks.male
) q05_tmp_Cust
;

