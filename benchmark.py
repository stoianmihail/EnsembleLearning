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
from classifiers.randomforest import RandomForest
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
    elif classifierName == "RandomForest":
      self.classifier = RandomForest()
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
          self.run("RandomForest")
          self.run("LogisticRegression")
          self.run("GaussianNB")
        
          # And separate
          print("-" * self.maxLen)
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
