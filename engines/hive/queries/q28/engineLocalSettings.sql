--If your query will not run with  hive.optimize.sampling.orderby=true because of a hive bug => disable orderby sampling locally in this file. 
--Default behaviour is to use whatever is default in ../engines/hive/conf/engineSettings.sql
--No need to disable orderby sampling globally for all queries and no need to modifiy the query files q??.sql.
set bigbench.hive.optimize.sampling.orderby=${hiveconf:bigbench.hive.optimize.sampling.orderby};
set bigbench.hive.optimize.sampling.orderby.number=${hiveconf:bigbench.hive.optimize.sampling.orderby.number};
set bigbench.hive.optimize.sampling.orderby.percent=${hiveconf:bigbench.hive.optimize.sampling.orderby.percent};