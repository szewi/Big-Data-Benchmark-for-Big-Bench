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

package io.bigdatabenchmark.v2.queries.q05

import java.io.{BufferedOutputStream, OutputStreamWriter}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression => LogisticRegressionSpark, LogisticRegressionModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{StructField, _}
// mllib is still needed for confusion matrix
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._

/**
 * Performs Logistic Regression analysis on data.
 * @author Christoph Bruecke <christoph.bruecke@bankmark.de>
 * @author Michael Frank <michael.frank@bankmark.de>
 * @version 1.0 24.02.2017 for spark >= 2
*/
object LogisticRegression {
  type OptionMap = Map[Symbol, String]

  val CSV_DELIMITER_OUTPUT = "," //same as used in hive engine result table to keep result format uniform


  def main (args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)

    if(args.isEmpty){
      println(usage)
      return
    }

    //parse command line options and provide a map of default values
    val options = parseOption(Map(
      'stepsize -> "1.0",
      'type -> "LBFGS",
      'iter -> "20",
      'lambda -> "0.0",
      'numClasses -> 2.toString,
      'numCorrections -> 10.toString,
      'convergenceTol -> 1e-5.toString,
      'fromHiveMetastore -> "true",
      'saveClassificationResult -> "true",
      'saveMetaInfo -> "true",
      'verbose -> "false",
      'csvInputDelimiter -> ","
    ), args.toList)
    println(s"Run LogisticRegression with options: $options")

    //pre-cleanup of output file/folder
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val path = new org.apache.hadoop.fs.Path(options('output))
    if (hdfs.exists(path)) {
      hdfs.delete(path, true)
    }

    val session = if (options('fromHiveMetastore).toBoolean) {
      SparkSession.builder().appName("q05_logisticRegression").enableHiveSupport().getOrCreate()
    } else {
      SparkSession.builder().appName("q05_logisticRegression").getOrCreate()
    }

    import session.implicits._

    // load the data and add label column
    val rawData = load(session, options)
    val average = rawData.select(mean($"clicks_in_category")).head().getDouble(0)
    val data = rawData
        .withColumn("label", when($"clicks_in_category" > average, 1.0d).otherwise(0.0d))

    //statistics of data (debug)
    if (options('verbose).toBoolean) {
      println("Average click count: " + average)
      println("Data Loaded: " + data.count() + " points.")
      println("Interested Count: " + data.filter($"label" === 0.0d).count())
      println("Uninterest Count: " + data.filter($"label" === 1.0d).count())
    }

    // merge all columns except "clicks_in_category" into one vector
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array(
        "college_education", "male",
        "clicks_in_1", "clicks_in_2", "clicks_in_3", "clicks_in_4", "clicks_in_5", "clicks_in_6", "clicks_in_7"))
      .setOutputCol("features")

    //train model with data using LBFGS LogisticRegression
    val logisticRegression = new LogisticRegressionSpark()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(options('iter).toInt)
      .setRegParam(options('lambda).toDouble)
      .setTol(options('convergenceTol).toDouble)

    val pipeline = new Pipeline().setStages(Array(vectorAssembler, logisticRegression))

    println("Training Model")
    val model = pipeline.fit(data)

    println("Predict with model")
    val predictions = model.transform(data)

    predictions.cache()

    if(options('verbose).toBoolean) {
      println("show first 10 lines of predictions")
      predictions.show(10)
    }

    //calculate Metadata about created model
    val logitModel = model.stages(model.stages.length - 1).asInstanceOf[LogisticRegressionModel]
    val summary = logitModel.summary.asInstanceOf[BinaryLogisticRegressionSummary]
    val multMetrics = new MulticlassMetrics(
      predictions.select($"prediction", $"label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    )

    val auc = summary.areaUnderROC
    val prec = multMetrics.precision
    val confMat = multMetrics.confusionMatrix

    val metaInformation =
    f"""Precision: $prec%.4f
         |AUC: $auc%.4f
         |Confusion Matrix:
         |$confMat
         |""".stripMargin

    println(metaInformation)

    //store prediction to HDFS
    if(options('saveClassificationResult).toBoolean) {
      println("save prediction to " + options('output))
      predictions.select($"label", $"prediction").write.csv(options('output))
    }

    //store to HDFS
    if(options('saveMetaInfo).toBoolean) {
      val metaInfoOutputPath = new Path(options('output), "metainfo.txt")
      println("save to " + metaInfoOutputPath)
      val out = new OutputStreamWriter (new BufferedOutputStream( hdfs.create(metaInfoOutputPath)))
      out.write(metaInformation)
      out.close()
    }

    session.stop()
  }

  /**
    * Loads input data from specified input location, either hive table or csv file.
    * @param session current SparkSession to be used for reading
    * @param options specifies the input location <code>'input</code>, whether to load from hive metastore
    *                <code>'fromHiveMetastore</code> and the csv input delimiter <code>'csvInputDelimiter</code>.
    * @return a DataFrame with the following schema
    *         <br/>
    *         {{{
    *           clicks_in_category BIGINT, -- used as label - number of clicks in specified category "q05_i_category"
    *           college_education  BIGINT, -- has college education [0,1]
    *           male               BIGINT, -- isMale [0,1]
    *           clicks_in_1        BIGINT, -- number of clicks in category id 1
    *           clicks_in_2        BIGINT, -- number of clicks in category id 2
    *           clicks_in_3        BIGINT, -- number of clicks in category id 3
    *           clicks_in_4        BIGINT, -- number of clicks in category id 4
    *           clicks_in_5        BIGINT, -- number of clicks in category id 5
    *           clicks_in_6        BIGINT, -- number of clicks in category id 6
    *           clicks_in_7        BIGINT  -- number of clicks in category id 7
    *         }}}
    */
  def load(session: SparkSession, options: OptionMap) : DataFrame = {
    if (options('fromHiveMetastore).toBoolean) {
      session.sql(
        s""" SELECT *
           | FROM ${options('input)}
         """.stripMargin)
    } else {
      val schema = StructType(Array(
        StructField("clicks_in_category", LongType, nullable = false),
        StructField("college_education", LongType, nullable = false),
        StructField("male", LongType, nullable = false),
        StructField("clicks_in_1", LongType, nullable = false),
        StructField("clicks_in_2", LongType, nullable = false),
        StructField("clicks_in_3", LongType, nullable = false),
        StructField("clicks_in_4", LongType, nullable = false),
        StructField("clicks_in_5", LongType, nullable = false),
        StructField("clicks_in_6", LongType, nullable = false),
        StructField("clicks_in_7", LongType, nullable = false)
      ))
      session.read.schema(schema).option("delimiter", options('csvInputDelimiter)).csv(options('input))
    }
  }

  /**
   * convertes row from dataFrame to LabeldPoint
   * @param row Input row
   * @param mean mean (threshold) to convert old labels to binary labels
   * @return
   */
  def transformRow(row: Row, mean: Double) : (Double, DenseVector) = {
    val doubleValues = (for (i <- 0 until row.size) yield {
      row.getLong(i).toDouble
    }).toArray

    val label = if (doubleValues(0) > mean) {1.0} else {0.0}
    (label, new DenseVector(doubleValues))
  }


  val usage : String =
    """
      |Options:
      |[-i  | --input <input dir> OR <database>.<table>]
      |[-o  | --output output folder]
      |[-d  | --csvInputDelimiter <delimiter> (only used if load from csv)]
      |[--type LBFGS|SGD]
      |[-it | --iterations iterations]
      |[-l  | --lambda regularizationParameter]
      |[-n  | --numClasses ]
      |[-t  | --convergenceTol ]
      |[-c  | --numCorrections (LBFGS only) ]
      |[-s  | --step-size size (SGD only)]
      |[--fromHiveMetastore true=load from hive table | false=load from csv file]
      |[--saveClassificationResult store classification result into HDFS
      |[--saveMetaInfo store metainfo about classification (cluster centers and clustering quality) into HDFS
      |[-v  | --verbose]
      |Defaults:
      |  step size: 1.0 (only used with --type sgd)
      |  type: LBFGS
      |  iterations: 20
      |  lambda: 0
      |  numClasses: 2
      |  numCorrections: 10
      |  convergenceTol: 1e-5.
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
      case "-o" :: value :: tail => parseOption(map ++ Map('output-> value), tail)
      case "--output" :: value :: tail => parseOption(map ++ Map('output-> value), tail)
      case "-d" :: value :: tail => parseOption(map ++ Map('csvInputDelimiter-> value), tail)
      case "--csvInputDelimiter" :: value :: tail => parseOption(map ++ Map('csvInputDelimiter-> value), tail)
      case "--type" :: value :: tail => parseOption(map ++ Map('type -> value), tail)
      case "-s" :: value :: tail => parseOption(map ++ Map('stepsize -> value), tail)
      case "--step-size" :: value :: tail => parseOption(map ++ Map('stepsize -> value), tail)
      case "-it" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "--iterations" :: value :: tail => parseOption(map ++ Map('iter -> value), tail)
      case "-l" :: value :: tail => parseOption(map ++ Map('lambda -> value), tail)
      case "--lambda" :: value :: tail => parseOption(map ++ Map('lambda -> value), tail)
      case "-n" :: value :: tail => parseOption(map ++ Map('numClasses -> value), tail)
      case "--numClasses" :: value :: tail => parseOption(map ++ Map('numClasses -> value), tail)
      case "-c" :: value :: tail => parseOption(map ++ Map('numCorrections -> value), tail)
      case "--numCorrections" :: value :: tail => parseOption(map ++ Map('numCorrections -> value), tail)
      case "-t" :: value :: tail => parseOption(map ++ Map('convergenceTol -> value), tail)
      case "--convergenceTol" :: value :: tail => parseOption(map ++ Map('convergenceTol -> value), tail)
      case "--fromHiveMetastore" :: value :: tail => parseOption(map ++ Map('fromHiveMetastore -> value), tail)
      case "--saveClassificationResult" :: value :: tail => parseOption(map ++ Map('saveClassificationResult -> value), tail)
      case "--saveMetaInfo" :: value :: tail => parseOption(map ++ Map('saveMetaInfo -> value), tail)
      case "-v" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case "--verbose" :: value :: tail => parseOption(map ++ Map('verbose -> value), tail)
      case option :: tail =>
        println("Bad Option " + option)
        println(usage)
        map
    }
  }
}