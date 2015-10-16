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

package io.bigdatabenchmark.v1.queries.q05

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.{SparkConf, SparkContext}
import io.bigdatabenchmark.v1.queries.OptionMap.OptionMap

/**
* Performs Logistic Regression analysis on data.
* @author Michael Frank <michael.frank@bankmark.de>
* @version 1.0 01.10.2015
*/
object LogisticRegression {

  val CSV_DELIMITER = " "

  def main (args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)

    /*- input table layout:
    label             BIGINT,--clicks_in_category
    college_education BIGINT,
    male              BIGINT,
    clicks_in_1       BIGINT,
    clicks_in_2       BIGINT,
    clicks_in_7       BIGINT,
    clicks_in_4       BIGINT,
    clicks_in_5       BIGINT,
    clicks_in_6       BIGINT
   */

    //parse command line options and provide a map of default values
    val options = parseOption(Map('stepsize -> "1.0",
      'type -> "LBFGS",
      'iter -> "20",
      'reg -> "0.0",
      'verbose -> "false",
      'fromMetastore -> "false"
    ), args.toList)
    val fromMetastoreTable = options('fromMetastore).toBoolean

    //pre-cleanup of output file/folder
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try { hdfs.delete(new org.apache.hadoop.fs.Path(options('of)), true) } catch { case _ : Throwable => { } }


    val conf = new SparkConf().setAppName("q05_logisticRegression")
    val sc = new SparkContext(conf)

    //load data and convert to RDD[LabeldPoint]
    val (average, points) = if (fromMetastoreTable) {
      loadFromMetastore(options, sc)
    } else {
      loadFromTextFile(options, sc)
    }

    //statistics of data (debug)
    if (options('verbose).toBoolean) {
      println("Average click count: " + average)
      println("Data Loaded: " + points.count() + " points.")
      println("Interested Count: " + points.filter(l => l.label == 1.0).count())
      println("Uninterest Count: " + points.filter(l => l.label == 0.0).count())
    }

    //train model with data using either SGD or LBFGS LogisticRegression
    println("Training Model")
    val model = getLogistcRegressionModel(options,points)

    println("Predict with model")
    val predictionAndLabels = points.map {
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    val binMetrics = new BinaryClassificationMetrics(predictionAndLabels)
    val multMetrics = new MulticlassMetrics(predictionAndLabels)

    val prec = multMetrics.precision
    val auc = binMetrics.areaUnderROC()
    val confMat = multMetrics.confusionMatrix

    //print and save result
    val result =
      s"""Precision: $prec
         |AUC: $auc
         |Confusion Matrix:
         |$confMat
       """.stripMargin

    println(result)

    println("save to " + options('of))
    sc.parallelize(List((result)), 1).saveAsTextFile(options('of))
    sc.stop()
  }


  def loadFromTextFile(options: OptionMap, sc: SparkContext): (Double, RDD[LabeledPoint]) = {
    println(s"load data from flatfile ${options('if)} ...")
    val csvData = sc.textFile(options('if))
    //csvData.cache() //don't know yet if caching is beneficial for runtime in this case

    //clicks are in first column, cols delimited by " "
    val averageClickCount = csvData.map(line => line.substring(0,line.indexOf(CSV_DELIMITER)).toDouble ).mean()

    val points: RDD[LabeledPoint] = csvData.map(line => csvLineToLP(line, averageClickCount)).cache()
    csvData.unpersist(false)
    (averageClickCount, points)
  }

  /**
   * load data from hive metastore table. !!not tested!!
   *
   * @param options uses the "-if" option as input for hive metastore table name in format: "<db>.<table>"
   * @param sc
   * @return
   */
  def loadFromMetastore(options: OptionMap, sc: SparkContext):  (Double, RDD[LabeledPoint]) = {
    println(s"load data from metastore table ${options('if)} ...")
    val sqlCtx = new HiveContext(sc)
    val inputTable = sqlCtx.table(options('if))
    if (options('verbose).toBoolean) {
      println(s"first 10 lines of ${options('if)}:")
      inputTable.show(10)
    }
    val firstRow = inputTable.select(avg(inputTable.col("label"))).head()
    println(s"firstRow: $firstRow")
    val average = firstRow.getDouble(0)
    val points: RDD[LabeledPoint] = inputTable.map(line => dataFrameRowToLP(line, average)).cache()
    (average, points)
  }


  def getLogistcRegressionModel(options: OptionMap, points:RDD[LabeledPoint]): LogisticRegressionModel = {
    val gradType = options('type)
    gradType match {
      case "SGD" =>
        val regression = new LogisticRegressionWithSGD()
        regression.optimizer.setStepSize(options('stepsize).toDouble).setNumIterations(options('iter).toInt).setRegParam(options('reg).toDouble)
        regression.run(points)
      case _ =>
        val regression = new LogisticRegressionWithLBFGS().setNumClasses(2)
        regression.optimizer.setNumIterations(options('iter).toInt).setNumCorrections(10).setConvergenceTol(1e-5).setRegParam(options('reg).toDouble)
        regression.run(points)
    }
  }

  /**
   * convertes row from dataFrame to LabeldPoint
   * @param line
   * @param mean
   * @return
   */
  def dataFrameRowToLP(line: Row, mean: Double): LabeledPoint = {
    val parsedToDouble = (for (i <- 0 to line.size) yield {
      line.getDouble(i)
    }).toArray

    val label = if (parsedToDouble(0) > mean) {1.0} else {0.0}
    LabeledPoint(label, Vectors.dense(parsedToDouble))
  }

  /**
   * Convertes a line of text, delimited by " "
   * @param s
   * @param mean
   * @return
   */
  def csvLineToLP(s: String, mean: Double): LabeledPoint = {
    val doubleValues = s.split(CSV_DELIMITER).map(_.toDouble)
    val label = if (doubleValues(0) > mean) {1.0} else {0.0}
    LabeledPoint(label, Vectors.dense(doubleValues))
  }


  val usage =
    """
      Options:
      [--fromMetastore true=load from hive table | false=load from csv file]
      [-if | --in-file inputfile/dir] [-of | --out-folder outputfolder]
      [-s | --step-size size] [--type LBFGS|SGD] [-i | --iterations iterations]
      [-l | --lambda regularizationParameter]
      [-v | --verbose]
      Defaults:
        step size: 1.0 (only used with --type sgd)
        type: LBFGS
        iterations: 20
        lambda: 0
    """.stripMargin

  def parseOption(map: OptionMap, args: List[String]) : OptionMap = {
    args match {
      case Nil => map
      case "--fromMetastore" :: value :: tail => parseOption(map ++ Map('fromMetastore -> value), tail)
      case "-if" :: value :: tail => parseOption(map ++ Map('if -> value), tail)
      case "-of" :: value :: tail => parseOption(map ++ Map('of -> value), tail)
      case "--in-file" :: value :: tail => parseOption(map ++ Map('if -> value), tail)
      case "--out-folder" :: value :: tail => parseOption(map ++ Map('of -> value), tail)
      case "-s" :: value :: tail => parseOption(map ++ Map('stepsize -> value), tail)
      case "--step-size" :: value :: tail => parseOption(map ++ Map('stepsize -> value), tail)
      case "--type" :: value :: tail => parseOption(map ++ Map('type -> value), tail)
      case "-i" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "--iterations" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "-l" :: value :: tail => parseOption(map ++ Map('reg -> value), tail)
      case "--lambda" :: value :: tail => parseOption(map ++ Map('reg -> value), tail)
      case "-v" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case "--verbose" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case option :: tail =>
        println("Bad Option " + option)
        println(usage)
        map
    }
  }
}