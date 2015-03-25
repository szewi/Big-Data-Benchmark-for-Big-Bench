--What is the ratio between the number of items sold over
--the internet in the morning (8 to 9am) to the number of items sold in the evening
--(7 to 8pm) of customers with a specified number of dependents. Consider only
--websites with a high amount of content.

-- Resources

--Result  --------------------------------------------------------------------
--keep result human readable

set hive.exec.compress.output=false;
set hive.exec.compress.output;
--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE} (
  am_pm_ratio DOUBLE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}';

-- Begin: the real query part
INSERT INTO TABLE ${hiveconf:RESULT_TABLE}
SELECT CAST(amc as double) / CAST(pmc as double) am_pm_ratio
FROM (
  SELECT COUNT(*) amc
  FROM web_sales ws
  JOIN household_demographics hd ON hd.hd_demo_sk = ws.ws_ship_hdemo_sk
  AND hd.hd_dep_count = ${hiveconf:q14_dependents}
  JOIN time_dim td ON td.t_time_sk = ws.ws_sold_time_sk
  AND td.t_hour >= ${hiveconf:q14_morning_startHour}
  AND td.t_hour <= ${hiveconf:q14_morning_endHour}
  JOIN web_page wp ON wp.wp_web_page_sk =ws.ws_web_page_sk
  AND wp.wp_char_count >= ${hiveconf:q14_content_len_min}
  AND wp.wp_char_count <= ${hiveconf:q14_content_len_max}
) at
JOIN (
  SELECT COUNT(*) pmc
  FROM web_sales ws
  JOIN household_demographics hd ON ws.ws_ship_hdemo_sk = hd.hd_demo_sk
  AND hd.hd_dep_count = ${hiveconf:q14_dependents}
  JOIN time_dim td ON  td.t_time_sk =ws.ws_sold_time_sk
  AND td.t_hour >= ${hiveconf:q14_evening_startHour}
  AND td.t_hour <= ${hiveconf:q14_evening_endHour}
  JOIN web_page wp ON wp.wp_web_page_sk = ws.ws_web_page_sk
  AND wp.wp_char_count >= ${hiveconf:q14_content_len_min}
  AND wp.wp_char_count <= ${hiveconf:q14_content_len_max}
) pt
CLUSTER BY am_pm_ratio
;
