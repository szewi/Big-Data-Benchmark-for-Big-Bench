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
CREATE TEMPORARY FUNCTION extract_sentiment AS 'de.bankmark.bigbench.queries.q10.SentimentUDF';

-- Result file configuration
set QUERY_NUM=q10;
set resultTableName=${hiveconf:QUERY_NUM}result;
set resultFile=${env:BIG_BENCH_HDFS_ABSOLUTE_QUERY_RESULT_DIR}/${hiveconf:resultTableName};

--Result  --------------------------------------------------------------------		
--kepp result human readable // not in this case, as we extract a sentiment for EVERY line in the reviews
--set hive.exec.compress.output=false;
--set hive.exec.compress.output;	
--CREATE RESULT TABLE. Store query result externaly in output_dir/qXXresult/
DROP TABLE IF EXISTS ${hiveconf:resultTableName};
CREATE TABLE ${hiveconf:resultTableName}
	ROW FORMAT DELIMITED FIELDS TERMINATED BY ','	LINES TERMINATED BY '\n'
	STORED AS TEXTFILE LOCATION '${hiveconf:resultFile}' 
AS
-- the real query part
  SELECT extract_sentiment(pr_item_sk,pr_review_content) AS (pr_item_sk, review_sentence, sentiment, sentiment_word)  
  FROM product_reviews;
