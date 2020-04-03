import os
import sys
import numpy as np
import math
import time

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from classifiers.baseline import Classifier
from classifiers.adaboost import AdaBoost
from classifiers.decisiontree import DecisionTree
from classifiers.randomforest import RandomForests
from classifiers.gradientboost import GradientBoost

# The Benchmark
from pmlb import fetch_data

# The dataclass type and its features
from dataclasses import dataclass

# The plotter
import matplotlib.pyplot as plt

@dataclass
class DataReader:
  useBenchmark: bool = True # if we use the PML benchmark (https://github.com/EpistasisLab/penn-ml-benchmarks)
  
  # The constructor
  def __init__(self, useBenchmark = True):
    self.useBenchmark = useBenchmark
  
  # Read the data
  def read(self, dataName):
    # Default value?
    if not self.useBenchmark:
      # Read custom data from the file 'dataName' and exclude header
      with open(dataName, 'r') as file:
        X = []
        y = []
        next(file)
        for line in file:
          args = line.split(',')
          size = len(args)
          curr = [float(x) for x in args[:(size - 1)]]
          label = int(args[size - 1])
          X.append(curr)
          y.append(label)
        return X, y
    else:
      # Read data from benchmark
      X, y = fetch_data(dataName, return_X_y = True, local_cache_dir = '.')
      if len(X) != len(y):
        print("x-axis and y-axis do not have the same size")
        sys.exit(1)
      if len(X) == 0:
        print("data is empty")
        sys.exit(1)
      
      # And remove the file
      os.remove(dataName + ".tsv.gz")
      return X, y

class Benchmark:
  def __init__(self, classificationType):
    self.classificationType = classificationType
    pass
  
  class FeaturePrinter(Classifier):
    def fit(self, X, y):
      self.preprocessing(X, y)
      print(self.types)
      pass
    
  def measureOverfitting(self, classifierName):
    if classifierName == "RandomForests":
      self.iterations = 200
      self.classifier = RandomForests(self.iterations)
      self.classifier.fit(self.train_X, self.train_y)
        
      print("done fitting")
        
      trainingScores = self.classifier.checkFitting(self.train_X, self.train_y)
      testScores = self.classifier.checkFitting(self.test_X, self.test_y)
      output = open("big_debug_rf.txt", "w")
      for index, scoreList in enumerate(trainingScores):
        for ptr, score in enumerate(scoreList):
          output.write(str(index + 1) + ":" + str(ptr + 1) + ":" + str(score) + "," + str(testScores[index][ptr]) + "\n")
      output.close()

  def computeConfusionMatrix(self):
    confusionMatrix = np.array([[0., 0.], [0., 0.]])
    for index, sample in enumerate(self.test_X):
      predicted = self.classifier.predict(sample)
      confusionMatrix[predicted][int(self.test_y[index] == self.classifier.shrinked[1])] += 1
    return confusionMatrix

  def measureROC(self, classifierName):
    if classifierName == "RandomForests":
      size = len(self.train_X[0])
      self.classifier = RandomForests()
      self.classifier.fit(self.train_X, self.train_y)
      
      xs = []
      ys = []
      output = open("debug_roc.txt", "w")
      for (randomSize, confusionMatrix) in self.classifier.matrices:
        sensitivity = confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0])
        specificity = confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]) 
        
        xs.append(1 - specificity)
        ys.append(sensitivity)
        print(str(randomSize) + " -> " + str(1 - specificity) + "," + str(sensitivity)) 
        output.write(str(randomSize) + ":" + str(1 - specificity) + "," + str(sensitivity) + "\n")
      print(xs)
      print(ys)
      output.close()
    pass

  def run(self, classifierName):
    if classifierName == "FeaturePrinter":
      self.classifier = self.FeaturePrinter()
      self.classifier.fit(self.train_X, self.train_y)
      return
    if classifierName == "AdaBoost":
      self.classifier = AdaBoost()
    elif classifierName == "GradientBoost":
      self.classifier = GradientBoost()
    elif classifierName == "DecisionTree":
      self.classifier = DecisionTree()
    elif classifierName == "RandomForests":
      self.classifier = RandomForests()
    elif classifierName == "LogisticRegression":
      self.classifier = LogisticRegression(max_iter=10000)
    elif classifierName == "GaussianNB":
      self.classifier = GaussianNB()
    fitStart = time.time()
    self.classifier.fit(self.train_X, self.train_y)
    fitStop = time.time()
    
    scoreStart = time.time()
    score = self.classifier.score(self.test_X, self.test_y)
    scoreStop = time.time()
    toPrint = classifierName + ": accuracy=" + "{0:.2f}".format(score * 100) + "%, fit_time=" + "{0:.3f}".format(fitStop - fitStart) + ", score_time=" + "{0:.3f}".format(scoreStop - scoreStart)
    print(toPrint)
    self.maxLen = max(self.maxLen, len(toPrint))
    pass
    
  def runAll(self):
    if self.classificationType == "binary_classification.txt":
      with open("binary_classification.txt", "r") as file:
        dataReader = DataReader()
        for dataName in file:
          # Read the data
          dataName = dataName.strip()
          X, y = dataReader.read(dataName)
          
          # Split it into training data and testing data
          self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y)
          
          self.maxLen = 0
          print("Data: " + str(dataName))
          print("Sizes: (train=" + str(len(self.train_X)) + ", test=" + str(len(self.test_X)) + ")")
         
          # And run the classifiers
          self.run("FeaturePrinter")
          self.run("AdaBoost")
          self.run("DecisionTree")
          self.run("GradientBoost")
          self.run("RandomForests")
          self.run("LogisticRegression")
          self.run("GaussianNB")
        
          # And separate
          print("-" * self.maxLen)
    elif self.classificationType == "test_roc.txt":
      with open("test_roc.txt", "r") as file:
        dataReader = DataReader()
        for dataName in file:
          # Read the data
          dataName = dataName.strip()
          X, y = dataReader.read(dataName)
          
          # Split it into training data and testing data
          self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y)
          print("Data: " + str(dataName))
          print("Sizes: (train=" + str(len(self.train_X)) + ", test=" + str(len(self.test_X)) + ")")
  
          self.measureROC("RandomForests")
    elif self.classificationType == "test_overfitting.txt":
      with open("test_overfitting.txt", "r") as file:
        dataReader = DataReader()
        for dataName in file:
          # Read the data
          dataName = dataName.strip()
          X, y = dataReader.read(dataName)
          
          # Split it into training data and testing data
          self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y)
          print("Data: " + str(dataName))
          print("Sizes: (train=" + str(len(self.train_X)) + ", test=" + str(len(self.test_X)) + ")")
  
          self.measureOverfitting("RandomForests")
    elif self.classificationType == "test":
      dataReader = DataReader(False)
      X, y = dataReader.read("google_dataset.csv")
      self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y)
      for index, sample in enumerate(self.train_X):
        print(str(index) + " : " + str(sample) + " -> " + str(self.train_y[index]))
      for index, sample in enumerate(self.test_X):
        print(str(sample) + " -> " + str(self.test_y[index]))
      self.maxLen = 0
      self.run("GradientBoost")
    pass

def main(classificationType):
  benchmark = Benchmark(classificationType)
  benchmark.runAll()
  pass
    
if __name__ == '__main__':
  main(sys.argv[1])
