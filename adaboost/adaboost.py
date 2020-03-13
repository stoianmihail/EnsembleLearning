import os
import sys
import numpy as np
import math

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# The Benchmark
from pmlb import fetch_data

# The dataclass type and its features
from dataclasses import dataclass
from typing import List

# The plotter
import matplotlib.pyplot as plt

@dataclass
class DataReader:
  useBenchmark: bool = True
  
  def __init__(self, useBenchmark = True):
    self.useBenchmark = useBenchmark
  
  def read(self, dataName):
    # Default value?
    if not self.useBenchmark:
      # Read data from the file 'dataName' and exclude header
      with open(dataName, 'r') as file:
        X = []
        Y = []
        next(file)
        for line in file:
          args = line.split(',')
          size = len(args)
          curr = [float(x) for x in args[:(size - 1)]]
          label = int(args[size - 1])
          X.append(curr)
          Y.append(label)
        return X, Y
    else:
      # Read data from benchmark
      X, Y = fetch_data(dataName, return_X_y = True, local_cache_dir = '.')
      if len(X) != len(Y):
        print("x-axis and y-axis do not have the same size")
        sys.exit(1)
      if len(X) == 0:
        print("data is empty")
        sys.exit(1)
      # And remove the file
      os.remove(dataName + ".tsv.gz")
      return X, Y
    
@dataclass
class Stump:
  index: int = 0
  error: float = 1     # Used for computing the minimum error 
  alpha: float = 1     # This stump can be the best one
  feature: float = 0
  threshold: float = 0
  side: float = 0
  PERM = np.array([[1, -1], [-1, 1]])

  def __repr__(self) -> str:
    return "Stump: " + str(self.index) + " alpha=" + str(self.alpha) + " feature=" + str(self.feature) + " threshold=" + str(self.threshold) + " side=" + str(self.side)

  def setParameters(self, error, feature, threshold, side) -> None:
    self.error = error
    self.feature = feature
    self.threshold = threshold
    self.side = side

  def getError(self) -> float:
    return self.error
  
  def computeAlpha(self) -> None:
    if not (self.error in [0, 1]):
      self.alpha = 0.5 * math.log((1 - self.error) / self.error)
  
  def updateWeights(self, weights, correctPositions):
    n = len(weights)
    expr = [0.5 / self.error, 0.5 / (1 - self.error)]
    for index in range(0, n):
      weights[index] *= expr[int(index in correctPositions)]
  
  def predict(self, sample) -> float:
    return self.alpha * self.PERM[self.side][int(sample[self.feature] <= self.threshold)]

@dataclass
class AdaBoost:
  learners: List[Stump]
  T: int = 100
  
  def __init__(self, T):
    if T <= 0:
      print("AdaBoost needs a positive number of iterations")
      sys.exit(1)
    self.T = T
  
  def preprocessing(self, X, Y):
    numFeatures = len(X[0])
    store = []
    for index in range(0, numFeatures):
      store.append(dict())
    n = len(X)
    for indexInData in range(0, n):
      x = X[indexInData]
      for indexInSample in range(0, numFeatures):
        if not (x[indexInSample] in store[indexInSample]):
          store[indexInSample][x[indexInSample]] = [[indexInData], []] if not Y[indexInData] else [[], [indexInData]]
        else:
          store[indexInSample][x[indexInSample]][Y[indexInData]].append(indexInData)
        
    for index in range(0, numFeatures):
      store[index] = sorted(store[index].items())
    return store
  
  def fit(self, X, Y):
    # Initialize the weights
    n = len(X)
    weights = np.array([1.0 / n] * n)

    numFeatures = len(X[0])
    store = self.preprocessing(X, Y)

    binClass = [[], []]
    for index in range(0, n):
      binClass[Y[index]].append(index)

    # And compute
    self.learners = []
    for index in range(0, self.T):
      tmp = weights[binClass[0]].sum()
      totalWeights = [tmp, 1 - tmp]
      learner = Stump()
      for index in range(0, numFeatures):
        partialSum = [0, 0]
        for (discreteValue, [minusList, plusList]) in store[index]:
          # print(str(discreteValue) + " -> " + str(minusList) + " : " + str(plusList)) 
          partialSum[0] += weights[minusList].sum()
          partialSum[1] += weights[plusList].sum()
        
          # Compute the sum of weights of the mispredicted samples.
          # We can compute only the error when, for the current feature,
          # the samples with a less or equal value receive a 'minus' classification,
          # since the other case around is symmetric.
          minusError = partialSum[1] + totalWeights[0] - partialSum[0]
          plusError = 1 - minusError
          
          # And update the learner
          if minusError < min(plusError, learner.getError()):
            learner.setParameters(minusError, index, discreteValue, 0)
          elif plusError < min(minusError, learner.getError()):
            learner.setParameters(plusError, index, discreteValue, 1)
      
      # Check if we can terminate earlier
      if not learner.getError():
        return learners
      # Take the positions of the samples, for which the values are greater than the current threshold
      greaterThan = list(filter(lambda pos: X[pos][learner.feature] > learner.threshold, range(0, n)))
      
      # Take the positions of the samples, for whcih the values are less or equal than the current threshold
      lessOrEqual = list(filter(lambda pos: X[pos][learner.feature] <= learner.threshold, range(0, n)))
      
      # Compute the positions which have been correctly predicted by the current weak learner
      correctPositions = list(filter(lambda pos: Y[pos] == learner.side, lessOrEqual)) + list(filter(lambda pos: Y[pos] != learner.side, greaterThan))
  
      # Compute alpha of learner and update weights
      learner.computeAlpha()
      learner.updateWeights(weights, correctPositions)
      self.learners.append(learner)
    pass

  def query(self, sample):
    H = 0
    for learner in self.learners:
      H += learner.predict(sample)
    return int(H > 0)
  
  def score(self, X, Y):
    if len(X) != len(Y):
      print("x-axis and y-axis of training data do not have the same size")
      sys.exit(1)
    n = len(X)
    if not n:
      return 1
    else:
      correct = 0
      for index in range(0, n):
        q = self.query(X[index])
        correct += int(q == Y[index])
      accuracy = float(correct / n)
      return accuracy

def benchmark(dataName, tuning):
  dataReader = DataReader(True)
  X, y = dataReader.read(dataName)
  train_X, test_X, train_y, test_y = train_test_split(X, y)

  print("train data has size=" + str(len(train_X)))
  print("test data has size=" + str(len(test_X)))

  adaBoost = AdaBoost(tuning)
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
  pass
  
def main(dataName, tuning):
  benchmark(dataName, tuning)
  pass
    
if __name__ == '__main__':
  main(sys.argv[1], int(sys.argv[2]))
