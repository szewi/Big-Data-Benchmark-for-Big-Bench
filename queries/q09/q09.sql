--Calculate the total sales by diferent types of customers
--(e.g., based on marital status, education status), sales price and diferent com-
--binations of state and sales profit.

-- Resources




--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}' 
AS
-- the real query part
SELECT SUM(ss1.ss_quantity) 
FROM store_sales ss1 
-- select date range
LEFT SEMI JOIN  date_dim dd 	ON (ss1.ss_sold_date_sk = dd.d_date_sk AND dd.d_year=${hiveconf:q09_year})
JOIN customer_address ca1 	ON ss1.ss_addr_sk = ca1.ca_address_sk 
JOIN store s 				ON s.s_store_sk = ss1.ss_store_sk
JOIN customer_demographics cd ON cd.cd_demo_sk = ss1.ss_cdemo_sk 
WHERE
(( 
	    cd.cd_marital_status = '${hiveconf:q09_part1_marital_status}' 
	AND cd.cd_education_status = '${hiveconf:q09_part1_education_status}' 
	AND ${hiveconf:q09_part1_sales_price_min} <= ss1.ss_sales_price 
	AND ss1.ss_sales_price <= ${hiveconf:q09_part1_sales_price_max}

)OR(
	    cd.cd_marital_status = '${hiveconf:q09_part2_marital_status}' 
	AND cd.cd_education_status = '${hiveconf:q09_part2_education_status}' 
	AND ${hiveconf:q09_part2_sales_price_min} <= ss1.ss_sales_price 
	AND ss1.ss_sales_price <= ${hiveconf:q09_part2_sales_price_max}

)OR(
	    cd.cd_marital_status = '${hiveconf:q09_part3_marital_status}' 
	AND cd.cd_education_status = '${hiveconf:q09_part3_education_status}' 
	AND ${hiveconf:q09_part3_sales_price_min} <= ss1.ss_sales_price 
	AND ss1.ss_sales_price <= ${hiveconf:q09_part3_sales_price_max}
))
AND
((
		ca1.ca_country = '${hiveconf:q09_part1_ca_country}' 
	AND  ca1.ca_state IN (${hiveconf:q09_part1_ca_state_IN}) 
	AND  ${hiveconf:q09_part1_net_profit_min} <= ss1.ss_net_profit 
	AND  ss1.ss_net_profit <= ${hiveconf:q09_part1_net_profit_max}
)OR(
		ca1.ca_country = '${hiveconf:q09_part2_ca_country}' 
	AND  ca1.ca_state IN (${hiveconf:q09_part2_ca_state_IN}) 
	AND  ${hiveconf:q09_part2_net_profit_min} <= ss1.ss_net_profit 
	AND  ss1.ss_net_profit <= ${hiveconf:q09_part2_net_profit_max}
)OR(
	ca1.ca_country = '${hiveconf:q09_part3_ca_country}' 
	AND  ca1.ca_state IN (${hiveconf:q09_part3_ca_state_IN}) 
	AND  ${hiveconf:q09_part3_net_profit_min} <= ss1.ss_net_profit 
	AND  ss1.ss_net_profit <= ${hiveconf:q09_part3_net_profit_max}
))  
;
