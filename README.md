UNDER DEVELOPMENT
======

This document is a development version and describes the BigBench installation and execution on our AWS machines.

##### Table of Contents
[Cluster Environment](#cluster-environment)

[Installation](#installation)

[Data Generation](#data-generation)

[Run Queries](#run-queries)

[Some Helpers](#some-helpers)

## Cluster Environment

**SSH**

make sure passwordless SSH is working between all nodes for you current user.

**Java**

Java 1.7 is required. 64 bit is recommended

**Hadoop**

* Hive
* Mahout


## Installation

On the AWS installation, the github repository is cloned into a folder "nfs":

```
cd
mkdir nfs
cd nfs
git clone https://<username>@github.com/intel-hadoop/Big-Bench.git
```

Edit .bashrc:

```
cd
vi .bashrc
```

and add these lines to source the environment setup file:

```
if [ -f ~/nfs/Big-Bench/setEnvVars ]; then
        . ~/nfs/Big-Bench/setEnvVars
fi
```

Either logout/login or source the environment script manually to set the required variables:

`source ~/nfs/Big-Bench/setEnvVars`

### Configuration

Check, if the hadoop related variables are correctly set in the environment file:

`vi ~/nfs/Big-Bench/setEnvVars`

**Important:**

If you changed something you must either logout/login or source the setEnvVars script manually to make you changes visible to the environment. e.g:
`source ~/nfs/Big-Bench/setEnvVars`


Major settings, Specify your cluster environment:

```
BIG_BENCH_HIVE_LIBS           most important: the hive-contrib.jar
BIG_BENCH_HADOOP_LIBS         most important: hadoop-hdfs.jar commons-logging.jar log4j.jar as well as several  other hadoop-* and commons-* jars
BIG_BENCH_HADOOP_LIBS_NATIVE  (optional but speeds up hdfs access)
BIG_BENCH_HADOOP_CONF         most important: core-site.xml and hdfs-site.xml
BIG_BENCH_HDFS_MOUNT_POINT    
BIG_BENCH_HDFS_NAMENODE
```
Minor settings:
```
BIG_BENCH_USER
BIG_BENCH_DATAGEN_DFS_REPLICATION  replication count used during generation of the big bench table
BIG_BENCH_DATAGEN_JVM_ENV          -Xmx750m is sufficient for Nodes with 2 CPU cores, remove or increase if your Nodes have more cores
```


Add the nodes on which PDGF should generate data into the nodes.txt file:

`vi $BIG_BENCH_BASH_SCRIPT_DIR/nodes.txt`

In this file, list all hosts, one per line:

```
bb-aws2
bb-aws3
bb-aws4
```

**Important:** As a temporary measure (until PDGF can be run as a hadoop job), the directory structure must be replicated onto all nodes. So either repeat the "git clone" on every node (make sure that the directory structure is the same) or export the ~/nfs folder as a nfs share and mount it on all nodes in the ec2-user's ~/nfs directory (this is what we did on the AWS nodes). As a shared medium eases the following steps significantly, that approach is strongly recommended.

## Data Generation
### First run
Before the first PDGF run, the end user license must be accepted once. Therefore, PDGF must be started:

`java -jar $BIG_BENCH_DATA_GENERATOR_DIR/pdgf.jar `

Pressing ENTER shows the license. The license must be accepted by entering 'y'. After that, the PDGF shell can be exited by pressing 'q':

```
By using this software you must first agree to our terms of use. press [ENTER] to show
...
Do you agree to this terms of use [Y/N]?
y

PDGF:> q
```

**Important:** The license must be accepted in every PDGF location. So if no shared medium is used, the license must be accepted in every PDGF copy on all nodes, or accepted prior to distribution onto the nodes.

### Distributed generation
To generate data on the cluster nodes, run this command:

`$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchClusterDataGen.sh`

**Important:** default settings assume 2 cores per compute node! (small amazon ec2 instance). If you start the bigBenchClusterDataGen on bigger machiens you will run into a `java.lang.OutOfMemoryError: GC overhead limit exceeded` error. To Avoid this, please adapt setEnvVars -> BIG_BENCH_DATAGEN_JVM_ENV if you have compute nodes with more CPU cores. In this case remove the argument: `-Xmx750m` 

The data are being generated directly into HDFS (into the benchmarks/bigbench/data/ directory, absolute HDFS path is /user/ec2-user/benchmarks/bigbench/data/).

Default HDFS replication count is 1 (data is onyl stored on the generating node). You can change this in the $BIG_BENCH_HOME/setEnvVars file by changing the variable
`BIG_BENCH_DATAGEN_DFS_REPLICATION=<Replication count>' as described in: [Configuration](#Configuration)

### Hive Population 
Hive must create its own metadata to be able to access the generated data. 
Hive population is done after the data generation with `bigBenchPopulateHive.sh`:

`$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchPopulateHive.sh`

In case you want/must renew the hive tables, simply run the command again.

## Run Queries
Run all queries sequentially:

`$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchRunQueries.sh`

Run one specific query with this command:

`$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchRunQuery.sh <querNum>`

e.g:

`$BIG_BENCH_BASH_SCRIPT_DIR/bigBenchRunQuery.sh 1`

## Some Helpers
**during setup of setEnvVars**

Where is my hive libs folders for BIG_BENCH_HIVE_LIBS?

`find / -name "hive-contrib.jar" 2> /dev/null`

Where is my hadoop libs folder for BIG_BENCH_HADOOP_LIBS?

`find / -name "hadoop-hdfs.jar" 2> /dev/null`

Where is my hdfs native libs folder for BIG_BENCH_HADOOP_LIBS_NATIVE?

`find / -name "libhadoop.so" 2> /dev/null`

Where is my core-site.xml/hdfs-site.xml  for BIG_BENCH_HADOOP_CONF (usually the one in /etc/hadoop/...):

`find / -name "hdfs-site.xml" 2> /dev/null`

What is my name node address for BIG_BENCH_HDFS_NAMENODE? 

Take a look into hdfs-site.xml and locate this property value:
``` 
<property>
    <name>dfs.namenode.servicerpc-address</name>
    <value>host.domain:8022</value>
</property>
```

How to mount hdfs? execute or take a look at:

`$BIG_BENCH_BASH_SCRIPT_DIR/mounthdfs.sh`

**during query execution**

suspect something went wrong? the bigBenchRunQuery.sh and bigBenchRunQueries.sh scripts write logs to:

`$BIG_BENCH_LOGS_DIR`

`$BIG_BENCH_BASH_SCRIPT_DIR/showQueryErrors.sh`
(searches in all logs/q??.log files for error strings)

`$BIG_BENCH_BASH_SCRIPT_DIR/showQueryErrors.sh <query num>` 
(searches only in query specific log file for error strings)

something went terrible wrong? want to abort all jobs?

`$BIG_BENCH_BASH_SCRIPT_DIR/killAllHadoopJobs.sh`
