/*
 * "INTEL CONFIDENTIAL" Copyright 2017 Intel Corporation All Rights
 * Reserved.
 *
 * The source code contained or described herein and all documents related
 * to the source code ("Material") are owned by Intel Corporation or its
 * suppliers or licensors. Title to the Material remains with Intel
 * Corporation or its suppliers and licensors. The Material contains trade
 * secrets and proprietary and confidential information of Intel or its
 * suppliers and licensors. The Material is protected by worldwide copyright
 * and trade secret laws and treaty provisions. No part of the Material may
 * be used, copied, reproduced, modified, published, uploaded, posted,
 * transmitted, distributed, or disclosed in any way without Intel's prior
 * express written permission.
 *
 * No license under any patent, copyright, trade secret or other
 * intellectual property right is granted to or conferred upon you by
 * disclosure or delivery of the Materials, either expressly, by
 * implication, inducement, estoppel or otherwise. Any license under such
 * intellectual property rights must be express and approved by Intel in
 * writing.
 */

package io.bigdatabenchmark.v2.queries

import java.io.{BufferedOutputStream, OutputStreamWriter}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Performs Kmeans clustering on data. (used by multiple queries)
  * @author Christoph Bruecke <christoph.bruecke@bankmark.de>
  * @version 1.0 22.03.2017 for spark >= 2
  */
object KMeansClustering {

  val CSV_DELIMITER_OUTPUT = "," //same as used in hive engine result table to keep result format uniform

  type OptionMap = Map[Symbol, String]

  var options: OptionMap=Map('iter -> "20",
    'externalInitClusters -> "",
    'fromHiveMetastore -> "true",
    'csvInputDelimiter -> ",",
    'saveClassificationResult -> "true",
    'saveMetaInfo -> "true",
    'verbose -> "false"
  )

  def main (args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)

    if(args.isEmpty){
      println(usage)
      return
    }

    //parse cmd line options and pass defaults for parameters
    options = parseOption(options, args.toList)
    println(s"Run kmeans clustering with options: $options")

    val appName = options('qnum) + "_KMeansClustering"
    val session = if (options('fromHiveMetastore).toBoolean) {
      SparkSession.builder().appName(appName).enableHiveSupport().getOrCreate()
    } else {
      SparkSession.builder().appName(appName).getOrCreate()
    }

    //pre-cleanup of output file/folder
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val path = new org.apache.hadoop.fs.Path(options('output))
    if (hdfs.exists(path)) {
      hdfs.delete(path, true)
    }

    import session.implicits._

    println(s"read vectors from ${options('input)}")
    //load data and convert to DataFrame (label, feature_1, feature_2, ...)
    val data : DataFrame = load(session, options)

    if (options('verbose).toBoolean) {
      println("first 10 line from input:")
      data.show(10)
    }

    // vectorize all columns except the first column
    // (a, b, c, d, e) => (a, [b, c, d, e])
    val assembler = new VectorAssembler()
      .setInputCols(data.columns.tail)
      .setOutputCol("features")

    val kmeans = new KMeans()
      .setFeaturesCol(assembler.getOutputCol)
      .setK(options('numclust).toInt)
      .setMaxIter(options('iter).toInt)
      .setInitMode("k-means||")
      .setSeed(1234567890)

    val pipeline = new Pipeline()
      .setStages(Array(assembler, kmeans))

    val model = pipeline.fit(data)

    // result of the transformation and clustering
    val clustering = model.transform(data)

    // get the KMeansModel for meta information
    val kMeansModel = model.stages(1).asInstanceOf[KMeansModel]

    // Evaluate clusterModel by computing Within Set Sum of Squared Errors
    val wssse = kMeansModel.computeCost(clustering)
    val clusterCenters = kMeansModel.clusterCenters

    //print and write clustering  metadata
    val metaInformation =
      s"""Clusters:
         |
         |Number of Clusters: ${clusterCenters.length}
         |WWGE: $wssse
         |${clusterCenters.mkString("\n")}
       """.stripMargin
    println(metaInformation)

    //create & save clustering result result: "<label>,<cluster_id>"
    if(options('saveClassificationResult).toBoolean) {
      println("save classification result to " + options('output) + " format: <serialKey>,<cluster>")
      clustering.select($"label", $"prediction").write.option("delimiter", ",").csv(options('output))
    }

    if(options('saveMetaInfo).toBoolean) {
      val metaInfoOutputPath = new Path(options('output), "metainfo.txt")
      println("save meta info to: " + metaInfoOutputPath)
      val out = new OutputStreamWriter(new BufferedOutputStream(hdfs.create(metaInfoOutputPath)))
      out.write(metaInformation)
      out.close()
    }

    session.stop()
  }

  /**
    * Reads the data from a hive table or csv file and returns a DataFrame with the results.
    * @param session SparkSession to be used for reading the data
    * @param options Options specifying the input path, whether to read from hive metastore, and the csv input
    *                delimiter.
    * @return DataFrame with a label column (scalar) and a feature columns (dense vector)
    */
  def load(session: SparkSession, options: OptionMap) : DataFrame = {
    val data = if (options('fromHiveMetastore).toBoolean) {
      session.sql(
        s""" SELECT *
           | FROM ${options('input)}
         """.stripMargin)
    } else {
      session.read.option("inferSchema", true).csv(options('input))
    }
    data.withColumnRenamed(data.columns.head, "label")
  }

  val usage : String =
    """
      |Options:
      |[-i  | --input <input dir> OR <database>.<table>]
      |[-o  | --output output folder]
      |[-d  | --csvInputDelimiter <delimiter> (only used if load from csv)]
      |[-ic | --initialClusters initialClustersFile (initial clusters prohibit --iterations > 1)]
      |[-c  | --num-clusters clusters]
      |[-it | --iterations iterations]
      |[-q  | --query-num number for job identification ]
      |[--fromHiveMetastore true=load from hive table | false=load from csv file]
      |[--saveClassificationResult store classification result into HDFS
      |[--saveMetaInfo store metainfo about classification (cluster centers and clustering quality) into HDFS
      |[-v  | --verbose]
      |Defaults:
      |  iterations: 20
      |  fromHiveMetastore: true
      |  saveClassificationResult: true
      |  saveMetaInfo: true
      |  verbose: false
    """.stripMargin

  def parseOption(map: OptionMap, args: List[String]) : OptionMap = {
    args match {
      case Nil => map
      case "-i" :: value :: tail => parseOption(map ++ Map('input -> value), tail)
      case "--input" :: value :: tail => parseOption(map ++ Map('input -> value), tail)
      case "-o" :: value :: tail => parseOption(map ++ Map('output -> value), tail)
      case "--output" :: value :: tail => parseOption(map ++ Map('output -> value), tail)
      case "-d" :: value :: tail => parseOption(map ++ Map('csvInputDelimiter-> value), tail)
      case "--csvInputDelimiter" :: value :: tail => parseOption(map ++ Map('csvInputDelimiter-> value), tail)
      case "-c" :: value :: tail => parseOption(map ++ Map('numclust -> value), tail)
      case "--num-clusters" :: value :: tail => parseOption(map ++ Map('numclust -> value), tail)
      case "-it" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "--iterations" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "-q" :: value :: tail => parseOption(map ++ Map('qnum -> value), tail)
      case "--query-num" :: value :: tail => parseOption(map ++ Map('qnum -> value), tail)
      case "--initialClustersFile"  :: value :: tail => parseOption(map ++ Map('externalInitClusters -> value), tail)
      case "-ic"  :: value :: tail => parseOption(map ++ Map('externalInitClusters -> value), tail)
      case "--fromHiveMetastore" :: value :: tail => parseOption(map ++ Map('fromHiveMetastore -> value), tail)
      case "--saveClassificationResult" :: value :: tail => parseOption(map ++ Map('saveClassificationResult -> value), tail)
      case "--saveMetaInfo" :: value :: tail => parseOption(map ++ Map('saveMetaInfo -> value), tail)
      case "-v" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case "--verbose" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case option :: optionTwo :: tail =>
        println("Bad Option " + option)
        println(usage)
        map
      case _ => map
    }
  }
}
