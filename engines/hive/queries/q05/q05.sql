--Build a model using logistic regression: based on existing users online
--activities and demographics, for a visitor to an online store, predict the visitors
--likelihood to be interested in a given category.

-- Hive 0.12 bug, hive ignores  'hive.mapred.local.mem' resulting in out of memory errors in map joins!
-- (more exactly: bug in Hadoop 2.2 where hadoop-env.cmd sets the -xmx parameter multiple times, effectively overriding the user set hive.mapred.local.mem setting. see: https://issues.apache.org/jira/browse/HADOOP-10245
-- There are 3 workarounds: 
-- 1) assign more memory to the local!! Hadoop JVM client via 'HADOOP_CLIENT_OPTS'. (Cloudera->hive service settings->gateway default group->Resourcemanagement->Java-Heap size)
--    Map-join child vm will inherit the parents jvm settings. 
-- 2) reduce "hive.smalltable.filesize" to ~1MB (depends on your cluster settings for the local JVM)
-- 3) turn off "hive.auto.convert.join" to prevent hive from converting the join to a mapjoin.
--- set hive.auto.convert.join;
--- set hive.auto.convert.join=false;
--- 10mb smalltable.filesize should be sufficient for all static tables of bigbench
--- set hive.mapjoin.smalltable.filesize;
--- set hive.mapjoin.smalltable.filesize=12144;
--- set hive.mapred.local.mem;
--- set hive.mapred.local.mem=2048;
--- set hive.mapjoin.localtask.max.memory.usage;
--- set hive.mapjoin.localtask.max.memory.usage=0.99;

-- Resources

--Result  --------------------------------------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

--CREATE RESULT TABLE. Store query result externally in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE} (
  c_customer_sk     STRING,
  college_education STRING,
  male              STRING,
  label             STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_DIR}';

-- the real query part

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
SELECT
  q05_tmp_Cust.c_customer_sk,
  q05_tmp_Cust.college_education,
  q05_tmp_Cust.male,
  CASE WHEN q05_tmp_Cust.clicks_in_category > 2 THEN 1 ELSE 0 END AS label
FROM (
  SELECT
    q05_tmp_cust_clicks.c_customer_sk AS c_customer_sk,
    q05_tmp_cust_clicks.college_education AS college_education,
    q05_tmp_cust_clicks.male AS male,
    SUM(
      CASE WHEN q05_tmp_cust_clicks.i_category = ${hiveconf:q05_i_category}
      THEN 1
      ELSE 0 END
    ) AS clicks_in_category
  FROM (
    SELECT
      ct.c_customer_sk AS c_customer_sk,
      CASE WHEN cdt.cd_education_status IN (${hiveconf:q05_cd_education_status_IN})
        THEN 1 ELSE 0 END AS college_education,
      CASE WHEN cdt.cd_gender = ${hiveconf:q05_cd_gender}
        THEN 1 ELSE 0 END AS male,
      it.i_category AS i_category
    FROM customer ct
    INNER JOIN customer_demographics cdt ON ct.c_current_cdemo_sk = cdt.cd_demo_sk
    INNER JOIN web_clickstreams wcst ON (wcst.wcs_user_sk = ct.c_customer_sk
       AND wcst.wcs_user_sk IS NOT NULL)
    INNER JOIN item it ON wcst.wcs_item_sk = it.i_item_sk
  ) q05_tmp_cust_clicks
  GROUP BY q05_tmp_cust_clicks.c_customer_sk,
    q05_tmp_cust_clicks.college_education,
    q05_tmp_cust_clicks.male
) q05_tmp_Cust
;
