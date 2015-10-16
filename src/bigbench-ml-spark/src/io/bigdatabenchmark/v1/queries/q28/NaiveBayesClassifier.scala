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

package io.bigdatabenchmark.v1.queries.q28

import io.bigdatabenchmark.v1.queries.OptionMap.OptionMap
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Success, Try}

/**
 * Performs NaiveBayes classification on data.
 * @author Michael Frank <michael.frank@bankmark.de>
 * @version 1.0 01.10.2015
 */
object NaiveBayesClassifier {

  def main (args: Array[String]) {

    Logger.getLogger("org").setLevel(Level.OFF)
    val options = parseOption(Map('splitRatio -> "",
                                  'outputTestingClassification -> "false"), args.toList)

    //pre-cleanup of output file/folder
    val hadoopConf = new org.apache.hadoop.conf.Configuration()
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try { hdfs.delete(new org.apache.hadoop.fs.Path(options('of)), true) } catch { case _ : Throwable => { } }

    val sc = new SparkContext(new SparkConf().setAppName("q28_nb_classifier"))
    val preSplit=options('splitRatio).isEmpty

    //testdata allready in two separate files trainging/testing? if not => split
    val (trainingData, testingData) = if(preSplit){
      println(s"Input is pre-split, read 'training' from: ${options('inputTrain)} and 'testing' from:  ${options('inputTest)}")
      println("Reading training data")
      val trainingData = createLabledPoints(options('inputTrain), sc)
      println("Reading testing data")
      val testingData = createLabledPoints(options('inputTest), sc)
      (trainingData, testingData)
    }else {
      println(s"Input not pre-split, read from: ${options('inputTrain)} and random split into 'training' and 'testing' with ratio: ${options('splitRatio)}")
      val points = createLabledPoints(options('inputTrain), sc)
      val ratio=options('splitRatio).toDouble
      val splits = points.randomSplit(Array(ratio,1.0-ratio), seed = 7623425465362456L)
      val trainingData = splits(0)
      val testingData = splits(1)
      (trainingData, testingData)
    }

    println("Training NaiveBayes model")
    val model = NaiveBayes.train(trainingData, lambda = 1.0)

    println("Testing NaiveBayes model")
    val pred = testingData.map(p => (model.predict(p.features), p.label)).cache()
    val multMetrics = new MulticlassMetrics(pred)

    val prec = multMetrics.precision
    val confMat = multMetrics.confusionMatrix

    //print and store
    val metaInformation=s"""Precision: $prec
        |Confusion Matrix:
        |$confMat
        |""".stripMargin

    println(s"save to ${options('of)}")
    println(metaInformation)

    if(options('outputTestingClassification).toBoolean) {
      pred.zip(testingData).map(p => List(createSentiment(p._1._1), createSentiment(p._1._2), createSentiment(p._2.label)).toSeq).saveAsTextFile(options('of))
    }

    //save meta information file into same directory
    val metaInfoOutputPath = new org.apache.hadoop.fs.Path(options('of),"metainfo.txt")
    val out = hdfs.create(metaInfoOutputPath)
    out.writeChars(metaInformation)
    out.close()

    sc.stop()
  }

  /**
   * IN: <pr_review_sk>\t<pr_rating>\t<pr_review_content>
   * OUT: (<pr_rating>\t<pr_review_content>)
   * @param inputLine
   * @return
   */
  def parseInputDataLine(inputLine:String): Try[(String,String)] ={

    val splits = inputLine.split("\t")
    val tr = util.Try{
      (splits(1), splits(2))
    }
    tr.failed.foreach(e => println("Skipping bad line: " + inputLine))
    tr
  }

  def createLabledPoints(input:String, sc: SparkContext): RDD[LabeledPoint] = {

    //parse input and remove bad lines
    val cleandInput = sc.textFile(input).map(parseInputDataLine).collect{case Success(tuple) => tuple }

    //println(seq.take(10).deep.mkString("Sample:\n==================\n", "\n" , "\n...\n==============="))

    //split review text into terms using delimiter " " (simplistic)
    val terms: RDD[(String, Seq[String])] = cleandInput.mapValues(_.split(" ").toSeq).cache()

    println(s"calculcate TF")
    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(terms.values)
    tf.cache()

    println(s"calculcate TFIDF 1/2 => IDF.fit(TF)")
    val idf = new IDF().fit(tf)
    println(s"calculcate TFIDF 2/2 => transform IDF to TFIDF")
    val tfidf = idf.transform(tf)
    tf.unpersist(false)

    println(s"Create Points")
    val points = tfidf.zip(terms).map(t => {
      new LabeledPoint(extractSentiment(t._2._1), t._1)
    })
    terms.unpersist(false)
    points
  }

  val usage =
    """
      Options:
      [-i | --inputTraining trainingData] [-t | --inputTesting testingData]  [-of | --out-folder outputfolder"]
      [-r  | --ratio ratio (if ratio e.g. 0.3 is provided, -ic is ignored and -it is split into training (e.g 0.3) and classification (e.g. 0.7))) ]
      Defaults:
    """.stripMargin

  def parseOption(map: OptionMap, args: List[String]) : OptionMap = {
    args match {
      case Nil => map
      case "-i" :: value :: tail => parseOption(map ++ Map('inputTrain -> value), tail)
      case "--inputTraining" :: value :: tail => parseOption(map ++ Map('inputTrain -> value), tail)
      case "-t" :: value :: tail => parseOption(map ++ Map('inputTest -> value), tail)
      case "--inputTesting" :: value :: tail => parseOption(map ++ Map('inputTest -> value), tail)
      case "-of" :: value :: tail => parseOption(map ++ Map('of -> value), tail)
      case "--out-folder" :: value :: tail => parseOption(map ++ Map('of -> value), tail)
      case "-r" :: value :: tail => parseOption(map ++ Map('splitRatio -> value), tail)
      case "--ratio" :: value :: tail => parseOption(map ++ Map('splitRatio -> value), tail)
      case "--outputTestingClassification"  :: value :: tail => parseOption(map ++ Map('outputTestingClassification -> value), tail)
      case option :: optionTwo :: tail =>
        println("Bad Option " + option)
        println(usage)
        map
    }
  }

  def extractSentiment(label: String): Double = {
    label.split("/")(0) match {
      case "NEG" => -1.0
      case "POS" => 1.0
      case _ => 0.0
    }
  }

  def createSentiment(value: Double): String = {
    value match {
      case -1.0  => "NEG"
      case 1.0  => "POS"
      case _ =>  "NEUT"
    }
  }
}
