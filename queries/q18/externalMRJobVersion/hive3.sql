
-- Resources
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-maxent-3.0.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/opennlp-tools-1.5.3.jar;
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;
CREATE TEMPORARY FUNCTION extract_sentiment AS 'de.bankmark.bigbench.queries.q18.SentimentUDF'
;
-- Result file configuration


--Result  --------------------------------------------------------------------    
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;  
--Prepare result storage
DROP TABLE IF EXISTS ${hiveconf:RESULT_TABLE};
CREATE TABLE ${hiveconf:RESULT_TABLE}
  ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n'
  STORED AS ${env:BIG_BENCH_hive_default_fileformat_result_table} LOCATION '${hiveconf:RESULT_DIR}'
AS
-- the real query
SELECT   
   s_store_name 
  ,review_date 
  ,review_sentence
  , sentiment
  , sentiment_word 
FROM (
  SELECT extract_sentiment( s_store_name, pr_review_date, pr_review_content) 
      AS ( s_store_name, review_date, review_sentence, sentiment, sentiment_word )  
  -- select product_reviews containing the name of a store. Consider only stores with flat or declining sales in 3 consecutive months.
  FROM 
  (
    SELECT   
       s_store_name
      ,pr_review_date
      ,pr_review_content
    
    FROM 
    (
        --select store_name for stores with flat or declining sales in 3 consecutive months.
        SELECT s_store_name
        FROM  store s
        JOIN ${hiveconf:TEMP_TABLE} c on s.s_store_sk = c.cat
        WHERE  c.slope < 0 
    ) tmp
    JOIN   product_reviews pr ON (true)
    where instr(pr.pr_review_content,  tmp.s_store_name) >0

  ) foo
)ss
WHERE   ss.sentiment = 'NEG'
;
