import os
import sys
import numpy as np
import math
import time

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from adaboost import AdaBoost
from decisiontree import DecisionTree

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
  
  def run(self, classifierName):
    if classifierName == "AdaBoost":
      self.classifier = AdaBoost()
    elif classifierName == "DecisionTree":
      self.classifier = DecisionTree()
    elif classifierName == "LogisticRegression":
      self.classifier = LogisticRegression()
    elif classifierName == "GaussianNB":
      self.classifier = GaussianNB()
    fitStart = time.time()
    self.classifier.fit(self.train_X, self.train_y)
    fitStop = time.time()
    
    scoreStart = time.time()
    score = self.classifier.score(self.test_X, self.test_y)
    scoreStop = time.time()
    toPrint = classifierName + ": accuracy=" + "{0:.3f}".format(score) + ", fit_time=" + "{0:.3f}".format(fitStop - fitStart) + ", score_time=" + "{0:.3f}".format(scoreStop - scoreStart)
    print(toPrint)
    self.maxLen = max(self.maxLen, len(toPrint))
    pass
    
  def runAll(self):
    if self.classificationType == "binary_classification.txt":
      with open("binary_classification.txt", "r") as file:
        dataReader = DataReader()
        for dataName in file:
          # Read the data
          X, y = dataReader.read(dataName.strip())
          
          # Split it into training data and testing data
          self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y)
          
          self.maxLen = 0
          print("Data: " + str(dataName.strip()))
          print("Sizes: (train=" + str(len(self.train_X)) + ", test=" + str(len(self.test_X)) + ")")
          
          # And run the classifiers
          self.run("AdaBoost")
          self.run("DecisionTree")
          self.run("LogisticRegression")
          self.run("GaussianNB")
          
          # And separate
          print("-" * self.maxLen)
    elif classificationType == "test":
      dataReader = DataReader(False)
    pass

def main(classificationType):
  benchmark = Benchmark(classificationType)
  benchmark.runAll()
  pass
    
if __name__ == '__main__':
  main(sys.argv[1])
