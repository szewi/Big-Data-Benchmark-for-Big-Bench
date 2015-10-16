/*
 * "INTEL CONFIDENTIAL" Copyright 2015 Intel Corporation All Rights
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

package io.bigdatabenchmark.v1.queries

import io.bigdatabenchmark.v1.queries.OptionMap.OptionMap
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Performs Kmeans clustering on data. (used by multiple queries)
 * @author Michael Frank <michael.frank@bankmark.de>
 * @version 1.0 01.10.2015
 */
object KMeansClustering {

  def main (args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)

    //parse cmd line options and pass defaults for parameters
    val options = parseOption(Map('iter -> "20",
                                  'ic -> "",
                                  'verbose -> "false"), args.toList)

    println(s"Run kmeans clustering with options: $options")
    val sc = new SparkContext(new SparkConf().setAppName(options('qnum) + "_KMeansClustering"))

    //pre-cleanup of output file/folder
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try { hdfs.delete(new org.apache.hadoop.fs.Path(options('of)), true) } catch { case _ : Throwable => { } }


    println(s"read vectors from ${options('if)}")
    val raw_input_data = sc.textFile(options('if))

    if (options('verbose).toBoolean) {
      println("first 10 line from input:")
      println(raw_input_data.take(10).deep.mkString("\n") + "\n...")
    }

    //convert input to dense vectors for kmeans
    val vectors = raw_input_data.map(s => Vectors.dense(s.split(" ").drop(1).map(_.toDouble)))

    //configure KMeans
    val means = new KMeans().setK(options('numclust).toInt).setMaxIterations(options('iter).toInt)
    val centersFromFile = !options('ic).isEmpty
    if(false && centersFromFile) {
      //intial external clusters only available since spark 1.4
      /*
      val initialCluster_txt=sc.textFile(options('ic))
      val initialVectors= initialCluster_txt.map(s => Vectors.dense(s.split(" ").drop(1).map(_.toDouble))).collect()
      val initialClusters =  new KMeansModel(initialVectors)
      means.setInitialModel(initialClusters)
       */
    } else {
      means.setInitializationMode(KMeans.RANDOM).setSeed(675234312453645L)
    }

    //run clustering
    val clusters = means.run(vectors)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val wwge = clusters.computeCost(vectors)

    //print and write result
    var centersArray: List[Object] = Nil
    centersArray = clusters.clusterCenters.toList ::: List("WWGE:" + wwge, "Number of Clusters: " + clusters.k, "") ::: centersArray
    println(centersArray.reverse.mkString("Clusters:\n", "\n",""))
    println("save to " + options('of))
    sc.parallelize(centersArray.reverse, 1).saveAsTextFile(options('of))
    sc.stop()
  }

  val usage =
    """
      Options:
      [-if | --in-file inputfile] [-of | --out-folder outputfolder]
      [-ic | --initialClusters initialClustersFile (initial clusters prohibit -i > 1)]
      [-c  | --num-clusters clusters] [-i | --iterations iterations]
      [-q  | --query-num number for job identification ]
      [-v  | --verbose]
      Defaults:
        iterations: 20
    """.stripMargin

  def parseOption(map: OptionMap, args: List[String]) : OptionMap = {
    args match {
      case Nil => map
      case "-if" :: value :: tail => parseOption(map ++ Map('if -> value), tail)
      case "-of" :: value :: tail => parseOption(map ++ Map('of -> value), tail)
      case "--in-file" :: value :: tail => parseOption(map ++ Map('if -> value), tail)
      case "--out-folder" :: value :: tail => parseOption(map ++ Map('of -> value), tail)
      case "-c" :: value :: tail => parseOption(map ++ Map('numclust -> value), tail)
      case "--num-clusters" :: value :: tail => parseOption(map ++ Map('numclust -> value), tail)
      case "-i" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "--iterations" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "-q" :: value :: tail => parseOption(map ++ Map('qnum -> value), tail)
      case "--query-num" :: value :: tail => parseOption(map ++ Map('qnum -> value), tail)
      case "--initialClustersFile"  :: value :: tail => parseOption(map ++ Map('ic -> value), tail)
      case "-ic"  :: value :: tail => parseOption(map ++ Map('ic -> value), tail)
      case "-v" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case "--verbose" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case option :: optionTwo :: tail =>
        println("Bad Option " + option)
        println(usage)
        map
    }
  }
}
