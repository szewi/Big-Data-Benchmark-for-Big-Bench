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

package io.bigdatabenchmark.v2.queries.q28

import java.io.{BufferedOutputStream, OutputStreamWriter}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, _}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
 * Performs NaiveBayes classification on data.
 * @author Christoph Bruecke <christoph.bruecke@bankmark.de>
 * @author Michael Frank <michael.frank@bankmark.de>
 * @version 1.0 23.02.2017 for spark >= 2.0
 */
object NaiveBayesClassifier {

  val CSV_DELIMITER_OUTPUT = "," //same as used in hive engine result table to keep result format uniform


  type OptionMap = Map[Symbol, String]

  var options: OptionMap=Map('splitRatio -> "",
                            'lambda -> "1.0",
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
    options = parseOption(options, args.toList)
    println(s"Run NaiveBayes with options: $options")

    //pre-cleanup of output file/folder
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    val path = new org.apache.hadoop.fs.Path(options('output))
    if (hdfs.exists(path)) {
      hdfs.delete(path, true)
    }

    val preSplit = options('splitRatio).isEmpty
    val delimiter = options('csvInputDelimiter)
    val trainFileOrTable = options('inputTrain)
    val testFileOrTable = options.getOrElse('inputTest, "")
    val fromHiveMetastore = options('fromHiveMetastore).toBoolean
    val verbose = options('verbose).toBoolean

    val session = if (fromHiveMetastore) {
      SparkSession.builder().appName("q28_nb_classifier").enableHiveSupport().getOrCreate()
    } else {
      SparkSession.builder().appName("q28_nb_classifier").getOrCreate()
    }

    // needed for `$` operator
    import session.implicits._

    // function that reads the data from the given string (table or file) and remove empty reviews
    val loadFun : (String => DataFrame)=
      str => load(str , fromHiveMetastore, delimiter, session).filter($"pr_review_content".isNotNull)

    // read, and split (if necessary), the training and testing data
    val (inputTrain, inputTest) = if (preSplit) {
      (loadFun(trainFileOrTable), loadFun(testFileOrTable))
    } else {
      val ratio = options('splitRatio).toDouble
      val df = loadFun(trainFileOrTable)
      val splits = df.randomSplit(Array(ratio, 1 - ratio), 0xDEADBEEF)
      (splits(0), splits(1))
    }

    // prepare the datasets
    val transformRating = udf((r : Int) => ratingToDoubleLabel(r))
    val trainingData = inputTrain.withColumn("sentiment", transformRating($"pr_rating")).cache()
    val testingData = inputTest.withColumn("sentiment", transformRating($"pr_rating")).cache()

    if (verbose) {
      println("training distribution")
      trainingData.groupBy($"sentiment").agg(count("*")).show()
      println("testing distribution")
      testingData.groupBy($"sentiment").agg(count("*")).show()
    }

    // set up pipeline
    val tokenizer = new Tokenizer()
      .setInputCol("pr_review_content")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("rawFeature")

    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("feature")

    val naiveBayes = new NaiveBayes()
      .setSmoothing(options('lambda).toDouble)
      .setFeaturesCol(idf.getOutputCol)
      .setLabelCol("sentiment")
      .setModelType("multinomial")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, idf, naiveBayes))

    // train the model (Tokenize -> TF/IDF -> Naive Bayes)
    val model = pipeline.fit(trainingData)

    println("Testing NaiveBayes model")
    // get the predictions
    val prediction : DataFrame = model.transform(testingData).cache()

    // calculate metrics
    val predictionAndLabel = prediction
      .select($"prediction", $"sentiment")
      .rdd.map({case Row(prediction: Double, sentiment: Double) => prediction -> sentiment})
    val multMetrics = new MulticlassMetrics(predictionAndLabel)

    val prec = multMetrics.precision
    val acc = multMetrics.accuracy
    val confMat = multMetrics.confusionMatrix

    //calculate Metadata about created model
    val metaInformation=
      s"""Precision: $prec
         |Accuracy: $acc
         |Confusion Matrix:
         |$confMat
         |""".stripMargin

    println(metaInformation)

    //predict based on created model and store prediction to HDFS
    if(options('saveClassificationResult).toBoolean) {
      println(s"save to ${options('output)}")
      //output review_sk, original_rating, classification_result_string
      val sentimentToString = udf((l : Double) => labelToString(l))
      prediction
        .select($"pr_review_sk", $"pr_rating", sentimentToString($"prediction"))
        .write.option("delimiter", CSV_DELIMITER_OUTPUT).csv(options('output))
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

  def load(input : String, isTable : Boolean, delimiter : String, session: SparkSession) : DataFrame = {
    if (isTable) {
      session.sql(s"SELECT pr_review_sk, pr_rating, pr_review_content FROM $input")
    } else {
      val reviewSchema = StructType(Array(
        StructField("pr_review_sk", LongType, nullable = false),
        StructField("pr_rating", IntegerType, nullable = false),
        StructField("pr_review_content", StringType, nullable = false)
      ))
      session.read.schema(reviewSchema).option("delimiter", delimiter).csv(input)
    }
  }

  val usage : String =
    """
      |Options:
      |[-i | --inputTraining <input dir> OR <database>.<table>]
      |[-t | --inputTesting --input <input dir> OR <database>.<table>]
      |[-o | --output outputfolder]
      |[-d | --csvInputDelimiter <delimiter> (only used if load from csv)]
      |[-r | --training-ratio ratio (if ratio e.g. 0.3 is provided, -t (testing) is ignored and -i (training) is split into training (e.g 0.3) and classification (e.g. 0.7))) ]
      |[-l | --lambda ]
      |[--fromHiveMetastore true=load from hive table | false=load from csv file]
      |[--saveClassificationResult store classification result into HDFS]
      |[--saveMetaInfo store metainfo about classification (cluster centers and clustering quality) into HDFS]
      |[-v  | --verbose]
      |Defaults:
      |  lambda: 1
      |  fromHiveMetastore: true
      |  saveClassificationResult: true
      |  saveMetaInfo: true
      |  verbose: false
    """.stripMargin

  def parseOption(map: OptionMap, args: List[String]) : OptionMap = {
    args match {
      case Nil => map
      case "-i" :: value :: tail => parseOption(map ++ Map('inputTrain -> value), tail)
      case "--inputTraining" :: value :: tail => parseOption(map ++ Map('inputTrain -> value), tail)
      case "-t" :: value :: tail => parseOption(map ++ Map('inputTest -> value), tail)
      case "--inputTesting" :: value :: tail => parseOption(map ++ Map('inputTest -> value), tail)
      case "-o" :: value :: tail => parseOption(map ++ Map('output -> value), tail)
      case "--output" :: value :: tail => parseOption(map ++ Map('output -> value), tail)
      case "-d" :: value :: tail => parseOption(map ++ Map('csvInputDelimiter-> value), tail)
      case "--csvInputDelimiter" :: value :: tail => parseOption(map ++ Map('csvInputDelimiter-> value), tail)
      case "-r" :: value :: tail => parseOption(map ++ Map('splitRatio -> value), tail)
      case "--training-ratio" :: value :: tail => parseOption(map ++ Map('splitRatio -> value), tail)
      case "-l" :: value :: tail => parseOption(map ++ Map('lambda -> value), tail)
      case "--lambda" :: value :: tail => parseOption(map ++ Map('lambda -> value), tail)
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

  def ratingToDoubleLabel(label: Int): Double = {
    label match {
      case 1 => 0.0
      case 2 => 0.0
      case 3 => 1.0
      case 4 => 2.0
      case 5 => 2.0
      case _ => 1.0
    }
  }

  def labelToString(value: Double): String = {
    value match {
      case 0.0  => "NEG"
      case 2.0  => "POS"
      case _ =>  "NEUT"
    }
  }
}
