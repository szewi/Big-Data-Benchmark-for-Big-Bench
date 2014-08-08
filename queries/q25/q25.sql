--Customer segmentation analysis: Customers are separated along the
--following key shopping dimensions: recency of last visit, frequency of visits and
--monetary amount. Use the store and online purchase data during a given year
--to compute.

-- Resources

-- Query parameters

-- ss_sold_date_sk > 2002-01-02
DROP TABLE IF EXISTS ${hiveconf:TEMP_TABLE};
CREATE TABLE ${hiveconf:TEMP_TABLE} (
  cid     BIGINT,
  oid     BIGINT,
  dateid  BIGINT,
  amount  DOUBLE
);

INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
SELECT
  ss_customer_sk AS cid,
  ss_ticket_number AS oid,
  ss_sold_date_sk AS dateid,
  SUM(ss_net_paid) AS amount
FROM store_sales ss
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_date > '${hiveconf:q25_date}'
AND ss_customer_sk IS NOT NULL
GROUP BY
  ss_customer_sk,
  ss_ticket_number,
  ss_sold_date_sk
;


INSERT INTO TABLE ${hiveconf:TEMP_TABLE}
SELECT
  ws_bill_customer_sk AS cid,
  ws_order_number AS oid,
  ws_sold_date_sk AS dateid,
  sum(ws_net_paid) AS amount
FROM web_sales ws
JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
WHERE d.d_date > '${hiveconf:q25_date}'
AND ws_bill_customer_sk IS NOT NULL
GROUP BY
  ws_bill_customer_sk,
  ws_order_number,
  ws_sold_date_sk
;

-------------------------------------------------------------------------------
--2003-01-02 == date_sk 37621

------ create input table for mahout --------------------------------------
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;

DROP TABLE IF EXISTS ${hiveconf:TEMP_RESULT_TABLE};
CREATE TABLE ${hiveconf:TEMP_RESULT_TABLE} (
  cid        INT,
  recency    INT,
  frequency  INT,
  totalspend INT)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' LINES TERMINATED BY '\n'
STORED AS TEXTFILE LOCATION '${hiveconf:TEMP_RESULT_DIR}';

INSERT INTO TABLE ${hiveconf:TEMP_RESULT_TABLE}
SELECT
  cid AS id,
  CASE WHEN 37621 - max(dateid) < 60 THEN 1.0 ELSE 0.0 END -- 37621 == 2003-01-02
    AS recency,
  count(oid) AS frequency,
  SUM(amount) AS totalspend
FROM ${hiveconf:TEMP_TABLE}
GROUP BY cid;


--- CLEANUP--------------------------------------------
DROP TABLE ${hiveconf:TEMP_TABLE};
