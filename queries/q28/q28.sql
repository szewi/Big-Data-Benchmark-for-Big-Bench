-- Global hive options (see: Big-Bench/setEnvVars)
set hive.exec.parallel=${env:BIG_BENCH_hive_exec_parallel};
set hive.exec.parallel.thread.number=${env:BIG_BENCH_hive_exec_parallel_thread_number};
set hive.exec.compress.intermediate=${env:BIG_BENCH_hive_exec_compress_intermediate};
set mapred.map.output.compression.codec=${env:BIG_BENCH_mapred_map_output_compression_codec};
set hive.exec.compress.output=${env:BIG_BENCH_hive_exec_compress_output};
set mapred.output.compression.codec=${env:BIG_BENCH_mapred_output_compression_codec};
set hive.default.fileformat=${env:BIG_BENCH_hive_default_fileformat};
set hive.optimize.mapjoin.mapreduce=${env:BIG_BENCH_hive_optimize_mapjoin_mapreduce};
set hive.optimize.bucketmapjoin=${env:BIG_BENCH_hive_optimize_bucketmapjoin};
set hive.optimize.bucketmapjoin.sortedmerge=${env:BIG_BENCH_hive_optimize_bucketmapjoin_sortedmerge};
set hive.auto.convert.join=${env:BIG_BENCH_hive_auto_convert_join};
set hive.auto.convert.sortmerge.join=${env:BIG_BENCH_hive_auto_convert_sortmerge_join};
set hive.auto.convert.sortmerge.join.noconditionaltask=${env:BIG_BENCH_hive_auto_convert_sortmerge_join_noconditionaltask};
set hive.optimize.ppd=${env:BIG_BENCH_hive_optimize_ppd};
set hive.optimize.index.filter=${env:BIG_BENCH_hive_optimize_index_filter};

--display settings
set hive.exec.parallel;
set hive.exec.parallel.thread.number;
set hive.exec.compress.intermediate;
set mapred.map.output.compression.codec;
set hive.exec.compress.output;
set mapred.output.compression.codec;
set hive.default.fileformat;
set hive.optimize.mapjoin.mapreduce;
set hive.optimize.bucketmapjoin;
set hive.optimize.bucketmapjoin.sortedmerge;
set hive.auto.convert.join;
set hive.auto.convert.sortmerge.join;
set hive.auto.convert.sortmerge.join.noconditionaltask;

-- Database
use ${env:BIG_BENCH_HIVE_DATABASE};

-- Resources
--ADD FILE ${env:BIG_BENCH_QUERIES_DIR}/q28/mapper_q28.py;

--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
DROP TABLE IF EXISTS q28t_raining;
CREATE EXTERNAL TABLE q28t_raining (pr_review_sk INT, pr_rating INT, pr_item_sk INT, pr_review_content STRING) 
       ROW FORMAT DELIMITED
       FIELDS TERMINATED BY '\t'
       LINES TERMINATED BY '\n'
       STORED AS TEXTFILE
       LOCATION '${hiveconf:MH_DIR}';

INSERT OVERWRITE TABLE q28t_raining
SELECT pr_review_sk,
    (case pr_review_rating
        when 1 then 'NEG' 
        when 2 then 'NEG'
        when 3 then 'NEU'
        when 4 then 'POS'
        when 5 then 'POS'
    end) AS pr_rating,
    pr_item_sk,
    pr_review_content
    from product_reviews
    where pmod(pr_review_sk, 5) in (1,2,3);


--Result  --------------------------------------------------------------------		
--keep result human readable
set hive.exec.compress.output=false;
set hive.exec.compress.output;
DROP TABLE IF EXISTS q28_testing;
CREATE TABLE q28_testing (pr_review_sk INT, pr_rating INT, pr_item_sk INT, pr_review_content STRING) 
       ROW FORMAT DELIMITED
       FIELDS TERMINATED BY '\t'
       LINES TERMINATED BY '\n'
       STORED AS TEXTFILE
       LOCATION '${hiveconf:MH_DIR}2';

INSERT OVERWRITE TABLE q28_testing
SELECT pr_review_sk,
    (case pr_review_rating
        when 1 then 'NEG' 
        when 2 then 'NEG'
        when 3 then 'NEU'
        when 4 then 'POS'
        when 5 then 'POS'
    end) AS pr_rating,
    pr_item_sk,
    pr_review_content
    from product_reviews
    where pmod(pr_review_sk, 5) in (0,4);

--SELECT review_sk, review_rating, item_sk, review_content
--  FROM (FROM product_reviews pr
--         MAP pr.pr_review_sk, pr.pr_review_rating, 
--             pr.pr_item_sk, pr.pr_review_content
--       USING 'python mapper_q28.py'
--          AS review_sk, review_rating, 
--             item_sk, review_content
--     CLUSTER BY review_sk) mo;

--    pr_review_sk,
--    (case pr_review_rating
--        when 1 then 'NEG' 
--        when 2 then 'NEG'
--        when 3 then 'NEU'
--        when 4 then 'POS'
--        when 5 then 'POS'
--    end) AS pr_rating,
--    pr_item_sk,
--    pr_review_content
--    from product_reviews;

-------------------------------------------------------------------------------

--CREATE TABLE a32_testingt
--AS SELECT 
--  pr_review_sk,
--    (case pr_review_rating
--        when 1 then 'NEG' 
--        when 2 then 'NEG'
--        when 3 then 'NEU'
--        when 4 then 'POS'
--        when 5 then 'POS'
--    end) AS pr_rating,
--    pr_review_content,
--    pr_item_sk
--    from product_reviews
--    where pmod(pr_review_sk, 5) in (0, 4);

--remaining: implement Text Classifire using q28.py
