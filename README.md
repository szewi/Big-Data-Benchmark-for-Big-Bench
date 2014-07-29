UNDER DEVELOPMENT -- contact bhaskar.gowda@intel.com for help in running the workload.

======

This document is a development version and describes the BigBench installation and execution on our AWS machines.

# Preparation

## Cluster Environment

**Java**

Java 1.7 is required. 64 bit is recommended

**Hadoop**

* Hive
* Mahout

## Installation

On the AWS installation, clone the github repository into a folder stored in $INSTALL_DIR:

```
export INSTALL_DIR="$HOME" # adapt this to your location
cd $INSTALL_DIR
git clone https://<username>@github.com/intel-hadoop/Big-Bench.git
```

## Configuration

Check if the hadoop related variables are correctly set in the environment file:

`vi "$INSTALL_DIR/Big-Bench/setEnvVars"`

Major settings, Specify your cluster environment:

```
BIG_BENCH_HADOOP_LIBS_NATIVE  (optional but speeds up hdfs access)
BIG_BENCH_HADOOP_CONF         most important: core-site.xml and hdfs-site.xml
```
Minor settings:
```
BIG_BENCH_USER
BIG_BENCH_DATAGEN_DFS_REPLICATION  replication count used during generation of the big bench table
BIG_BENCH_DATAGEN_JVM_ENV          -Xmx750m is sufficient for Nodes with 2 CPU cores, remove or increase if your Nodes have more cores
```
# Run the workload

There are two different methods for running the workload: use the driver to simply perform a complete benchmark run or use the bash scripts to do partial tests. As the driver calls the bash scripts internally, both methods yield the same results.

## Common hints

The following paragraphs are important for both methods.

### Accept license

When running the data generator for the first time, the user must accept its license:

```
By using this software you must first agree to our terms of use. Press [ENTER] to show them
... # license is displayed
If you have read and agree to these terms of use, please type (uppercase!): YES and press [ENTER]
YES
```

### PDGF
The data are being generated directly into HDFS (into the $BIG_BENCH_HDFS_RELATIVE_INIT_DATA_DIR directory, absolute HDFS path is $BIG_BENCH_HDFS_ABSOLUTE_INIT_DATA_DIR).

Default HDFS replication count is 1 (data is only stored on the generating node). You can change this in the $BIG_BENCH_HOME/setEnvVars file by changing the variable
`BIG_BENCH_DATAGEN_DFS_REPLICATION=<Replication count>'.

## Using the BigBench driver

The BigBench driver is started with a script. To show all available options, you can call the help first:
```
"$INSTALL_DIR/scripts/bigBench runBenchmark -h
```

### Quick start

If a complete benchmark run should be performed and no data were generated previously, this is the command which should be executed:

```
"$INSTALL_DIR/scripts/bigBench runBenchmark -m <number of map tasks for data generation> -f <scale factor of dataset> -s <number of parallel streams in the throughput test>
```

This command will generate data, run the load-, power- and throughput-test and calculate the BigBench result.

So a complete benchmark run with all stages can be done by running (e.g., 4 map tasks, scale factor 100, 2 streams):
```
"$INSTALL_DIR/scripts/bigBench runBenchmark -m 4 -f 100 -s 2
```

After the benchmark finished, two log files are written: BigBenchResult.txt (which contains the driver's sysout messages) as well as BigBenchTimes.csv (which contains all measured timestamps/durations). The log directory can be specified with the -l option, it defaults to the BigBench's log dir ($BIG_BENCH_LOGS_DIR).

### More detailed explanation

There are four phases the driver traverses (only three are benchmarked though): data generation, load test, power test and throughput test. The driver has a clean option (-c) which does not run the benchmark but rather cleans the environment from previous runs (if for some reason all generated data should be cleaned).

#### Data generation

The data generation phase is not benchmarked by BigBench. The driver can skip this phase by setting "-sd". Skipping this phase is a good idea if data were already generated previously and the complete benchmark should be repeated with the same dataset size. In that case, generting data is not necessary as PDGF would generate the exact same data as in the previous run. If data generation is not skipped, two other options must be provided to the driver: "-m" sets the number of map tasks for PDGF's data generation, "-f" sets the scale factor determining the dataset size (1 scale factor equals 1 GiB). If "-c" is set and "-sd" is not set, the dataset directory in HDFS will be deleted.

#### Load test

Population of the hive metastore is the first phase that is benchmarked by BigBench. This phase can be skipped by providing "-sl" as an option. Re-populating the metastore is technically only necessary if the dataset has changed. Nevertheless, metastore population is part of the benchmark, so if this phase is skipped then no BigBench result can be computed. If "-c" is set and "-sl" is not set, the tables of the dataset in the metastore will be dropped (however it does not influence the query results from later tests).


#### Power test

This is the second phase that is benchmarked by BigBench. All queries run sequentially in one stream. The phase can be skipped with the option "-sp". Setting "-c" (and not "-sp") cleans previous power-test's results in the hive metastore tables and HDFS directories.

#### Throughput test

The throughput test is the last benchmark phase. All queries run in parallel streams in different order. The phase can be skipped with "-st". If this phase is not skipped, "-s" is a required option because that sets the number of parallel streams used in this phase. As in the other phases, setting "-c" (and not "-st") cleans the thoughput-test's results in the hive metastore and the HDFS directories.

## Using the bigBench bash script

The driver internally calls the $BIG_BENCH_BASH_SCRIPT_DIR/bigBench bash script along with a module name. So every step the driver performs (apart from the more complicated "query mixing" and multi-stream execution logic) can be run manually by executing this script with the proper options.

### Overview

The general syntax for the bigBench script is:
```
"$INSTALL_DIR/scripts/bigBench [global options] moduleName [module options]
```

At the moment there is only one module which processes module options itself, namely runBenchmark. All other modules currently do NO option processing. They rely on bigBench for option processing. Therefore when not running the runBenchmark module, global options must be specified.

All available options as well as all found modules can be listed by calling the script help:
```
"$INSTALL_DIR/scripts/bigBench -h
```

### Available options
* -b: This option chooses which binary will be used for the benchmark. WARNING: support for choosing other binaries than "hive" is implemented, but hive is the only tested binary. In fact, no other binary works so far. DO NOT USE THAT OPTION
* -d: Some more complex queries are split into multiple internal parts. This option chooses which internal query part will be executed. This is a developer only option. ONLY USE IF YOU KNOW WHAT YOU ARE DOING
* -f: The scale factor for PDGF. It is used by the clusterDataGen and hadoopDataGen modules
* -h: Show help
* -m: The map tasks used for data generation. It is used by the clusterDataGen and hadoopDataGen modules
* -p: The benchmark phase to use. It is necessary if subsequent query runs should not overwrite results of previous queries. The driver internally uses POWER_TEST_IN_PROGRESS, THROUGHPUT_TEST_FIRST_QUERY_RUN_IN_PROGRESS and THROUGHPUT_TEST_SECOND_QUERY_RUN_IN_PROGRESS. The default value when not providing this option is RUN_QUERY
* -q: Defines the query number to be executed
* -s: This option defines the number of parallel streams to use. It is only of any use with the runQueryInParallel module
* -t: Sets the stream number of the current query. This option is important so that one query can run multiple times in parallel without interfering with other instances
* -v: Use the provided file as initial metastore population script
* -w: Use the provided file as metastore refresh script
* -y: Use the provided file for custom query parameters
* -z: Use the provided file for custom hive settings

### Modules usage examples

* cleanData: cleans the dataset directory in HDFS. This module is automatically run by the data generator module to remove the dataset from the HDFS.
```
"$INSTALL_DIR/scripts/bigBench cleanData
```

* cleanMetastore: cleans the metastore dataset tables.
```
"$INSTALL_DIR/scripts/bigBench [-z <hive settings>] cleanMetastore
```

* cleanQueries: cleans all metastore tables and result directories in HDFS for all 30 queries. This module works as a wrapper for cleanQuery and does not work if "-q" is set as option.
```
"$INSTALL_DIR/scripts/bigBench [-p <benchmark phase>] [-t <stream number] [-z <hive settings>] cleanQueries
```

* cleanQuery: cleans metastore tables and result directories in HDFS for one query. Needs the query number to be set.
```
"$INSTALL_DIR/scripts/bigBench [-p <benchmark phase>] -q <query number> [-t <stream number] [-z <hive settings>] cleanQuery
```

* clusterDataGen: generates data using ssh on all defined nodes. This module is deprecated. Do not use it.

* hadoopDataGen: generates data using a hadoop job. Needs the map tasks and scale factor options.
```
"$INSTALL_DIR/scripts/bigBench -m <map tasks> -f <scale factor> hadoopDataGen
```

* populateMetastore: populates the metastore with the dataset tables.
```
"$INSTALL_DIR/scripts/bigBench [-v <population script>] [-z <hive settings>] populateMetastore
```

* refreshMetastore: refreshes the metastore with the refresh dataset.
```
"$INSTALL_DIR/scripts/bigBench [-w <refresh script>] [-z <hive settings>] refreshMetastore
```

* runBenchmark: runs the driver. This module parses its options itself. For details look at the driver usage section above.
```
"$INSTALL_DIR/scripts/bigBench runBenchmark [driver options]
```

* runQueries: runs all 30 queries sequentially. This module works as a wrapper for runQuery and does not work if "-q" is set as option.
```
"$INSTALL_DIR/scripts/bigBench [-p <benchmark phase>] [-t <stream number] [-z <hive settings>] runQueries
```

* runQuery: runs one query. Needs the query number to be set.
```
"$INSTALL_DIR/scripts/bigBench [-p <benchmark phase>] -q <query number> [-t <stream number] [-z <hive settings>] runQuery
```

* runQueryInParallel: runs one query on multiple parallel streams. This module is a wrapper for runQuery. Needs the query number ("-q") and total number of streams ("-s") to be set and the stream number ("-t") to be unset.
```
"$INSTALL_DIR/scripts/bigBench [-p <benchmark phase>] -q <query number> -s <number of parallel streams> [-z <hive settings>] runQueryInParallel
```

* showErrors: parses query errors in the log files after query runs.
```
"$INSTALL_DIR/scripts/bigBench showErrors
```

* showTimes: parses execution times in the log files after query runs.
```
"$INSTALL_DIR/scripts/bigBench showTimes
```

* zipQueryLogs: generates a zip file of all logs in the logs directory. It is run by the driver after each complete benchmark run. Subsequent runs override the old log files. A zip archive is created to save them before being overwritten.
```
"$INSTALL_DIR/scripts/bigBench zipQueryLogs
```
