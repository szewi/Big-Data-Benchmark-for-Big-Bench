-- Global hive options (see: Big-Bench/setEnvVars)
set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources
ADD JAR ${env:BIG_BENCH_QUERIES_DIR}/Resources/bigbenchqueriesmr.jar;
CREATE TEMPORARY FUNCTION extract_sentiment AS 'de.bankmark.bigbench.queries.q18.SentimentUDF'
;
-- Result file configuration
set QUERY_NUM=q18;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};

DROP TABLE IF EXISTS q18_storeSentiments;
CREATE TABLE q18_storeSentiments
AS
SELECT 	extract_sentiment(s_store_sk, pr_item_sk, pr_review_date, pr_review_content) 
	AS (store_sk, item_sk, review_date, review_sentence, sentiment, sentiment_word )	

FROM 
(
	SELECT 	 
		s_store_sk
		,pr_item_sk
		,pr_review_date
		,pr_review_content
		
	FROM 
	(
		  SELECT s_store_sk, s_store_name
		  FROM 	store s
		  JOIN q18_store_coefficient c on  s.s_store_sk = c.cat
		  WHERE  c.slope < 0 
	) tmp
	JOIN   product_reviews pr ON (true)
	where instr(pr.pr_review_content,  tmp.s_store_name) >0

) foo;
 	

--Result  --------------------------------------------------------------------		
--kepp result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;	
--Prepare result storage
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
	ROW FORMAT DELIMITED FIELDS TERMINATED BY ','	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE LOCATION '${hiveconf:resultFile}' 
AS
-- the real query
SELECT 	 st.s_store_name 
	,review_date 
	,review_sentence
	, sentiment
	, sentiment_word 
FROM q18_storeSentiments ss
JOIN store st on ss.store_sk  = st.s_store_sk 
AND   ss.sentiment = 'NEG'
;


DROP TABLE IF EXISTS q18_storeSentiments;











