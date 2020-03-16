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

def benchmark(dataName):
  dataReader = DataReader()
  X, y = dataReader.read(dataName)
  train_X, test_X, train_y, test_y = train_test_split(X, y)

  print("Train data: " + str(len(train_X)))
  print("Test data: " + str(len(test_X)))
    
  if False:
    adaBoost = AdaBoost()
    logit = LogisticRegression()
    gnb = GaussianNB()

    adaBoost.fit(train_X, train_y)
    logit.fit(train_X, train_y)
    gnb.fit(train_X, train_y)

    adaBoostScore = adaBoost.score(test_X, test_y)
    logitScore = logit.score(test_X, test_y)
    gnbScore = gnb.score(test_X, test_y)

    print("adaBoost=" + str(adaBoostScore))
    print("logic=" + str(logitScore))
    print("gauss=" + str(gnbScore))
  else:
    decisionTree = DecisionTree()
    decisionTree.fit(train_X, train_y)
  pass
  
def main(dataName):
  benchmark(dataName)
  pass
    
if __name__ == '__main__':
  main(sys.argv[1])
