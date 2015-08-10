--"INTEL CONFIDENTIAL"
--Copyright 2015  Intel Corporation All Rights Reserved.
--
--The source code contained or described herein and all documents related to the source code ("Material") are owned by Intel Corporation or its suppliers or licensors. Title to the Material remains with Intel Corporation or its suppliers and licensors. The Material contains trade secrets and proprietary and confidential information of Intel or its suppliers and licensors. The Material is protected by worldwide copyright and trade secret laws and treaty provisions. No part of the Material may be used, copied, reproduced, modified, published, uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's prior express written permission.
--
--No license under any patent, copyright, trade secret or other intellectual property right is granted to or conferred upon you by disclosure or delivery of the Materials, either expressly, by implication, inducement, estoppel or otherwise. Any license under such intellectual property rights must be express and approved by Intel in writing.

--based on tpc-ds q4
--Find customers who spend more money via web than in
--stores for a given year. Report customers first name, last name, their country of
--origin and identify if they are preferred customer.

-- Resources



-- Part 1 helper table(s) --------------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};

-- customer store sales
CREATE VIEW ${hiveconf:TEMP_TABLE1} AS
select ss_customer_sk AS customer_sk,
       sum( case when (d_year = ${hiveconf:q06_YEAR})   THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2)  ELSE 0 END) first_year_total,
       sum( case when (d_year = ${hiveconf:q06_YEAR}+1) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2)  ELSE 0 END) second_year_total
 from store_sales
     ,date_dim
 where ss_sold_date_sk = d_date_sk
   and d_year between ${hiveconf:q06_YEAR} and ${hiveconf:q06_YEAR} +1
 group by ss_customer_sk
;

-- customer web sales
CREATE  VIEW ${hiveconf:TEMP_TABLE2} AS
select ws_bill_customer_sk AS customer_sk ,
       sum( case when (d_year = ${hiveconf:q06_YEAR})   THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2)   ELSE 0 END) first_year_total,
       sum( case when (d_year = ${hiveconf:q06_YEAR}+1) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2)   ELSE 0 END) second_year_total
 from web_sales
     ,date_dim
 where ws_sold_date_sk = d_date_sk
   and d_year between ${hiveconf:q06_YEAR} and ${hiveconf:q06_YEAR} +1
 group by ws_bill_customer_sk

;         


--Part2: self-joins

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
    c_customer_sk BIGINT,
    c_first_name STRING,
    c_last_name STRING,
    c_preferred_cust_flag STRING,
    c_birth_country STRING,
    c_login STRING,
    c_email_address STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT
      c_customer_sk,
      c_first_name,
      c_last_name,
      c_preferred_cust_flag,
      c_birth_country,
      c_login,
      c_email_address
from ${hiveconf:TEMP_TABLE1} store ,
     ${hiveconf:TEMP_TABLE2} web ,
     customer c
where store.customer_sk = web.customer_sk
and   web.customer_sk =c_customer_sk
and   case when web.first_year_total > 0 then web.second_year_total / web.first_year_total else null end
       > case when store.first_year_total > 0 then store.second_year_total / store.first_year_total else null end
ORDER BY
  c_customer_sk,
  c_first_name,
  c_last_name,
  c_preferred_cust_flag,
  c_birth_country,
  c_login
LIMIT ${hiveconf:q06_LIMIT};

---Cleanup-------------------------------------------------------------------
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE1};
DROP VIEW IF EXISTS ${hiveconf:TEMP_TABLE2};
