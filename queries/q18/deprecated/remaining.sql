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
-------------------------------------------------------------------------------
--------------------------------REMAINING--------------------------------------
-------------------------------------------------------------------------------
-- identify stores with declining sales---
-- and extract sentiment regarding the stores with declining sales, if any
select s_store_name, pr_review_date,out_content, out_polarity, out_sentiment_words
from ExtractSentiment
(
	on (
		select 
			s_store_name
			,pr_review_content
			,pr_review_date
		from 
			store_coefficient c
			,store100 s 
			,product_reviews100
		where 
			c.slope < 0 
			and s.s_store_sk = c.store
			and pr_review_content like '%'||s_store_name||'%'
	)
	text_column('pr_review_content')
	model('dictionary')
	level('DOCUMENT')
	accumulate('s_store_name','pr_review_date')
)
where out_polarity = 'NEG';

drop view store_coefficient;
drop view time_series_store;

end;
