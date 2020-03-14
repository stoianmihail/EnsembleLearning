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
    
@dataclass
class Stump:
  error: float = 1     # Used for computing the minimum error 
  alpha: float = 1     # This stump can be the best one
  feature: float = 0
  threshold: float = 0
  side: int = 0        # binary value: 0 -> less or equal values are "-", 1 -> otherwise
  PERM = np.array([[1, -1], [-1, 1]])

  # Print the stump
  def __repr__(self) -> str:
    return "Stump: alpha=" + str(self.alpha) + " feature=" + str(self.feature) + " threshold=" + str(self.threshold) + " side=" + str(self.side)

  # Set the parameters when updating the best stump
  def setParameters(self, error, feature, threshold, side) -> None:
    self.error = error
    self.feature = feature
    self.threshold = threshold
    self.side = side

  def getError(self) -> float:
    return self.error
  
  # Compute alpha which reduces the error bound of the total error
  def computeAlpha(self) -> None:
    if not (self.error in [0, 1]):
      self.alpha = 0.5 * math.log((1 - self.error) / self.error)
  
  # Return the prediction of this stump
  def predict(self, sample) -> int:
    return self.PERM[self.side][int(sample[self.feature] <= self.threshold)]
  
  # Rescale the weights, as described in MIT youtube video (https://www.youtube.com/watch?v=UHBmv7qCey4)
  def updateWeights(self, X, y, weights):
    # First compute alpha
    self.computeAlpha()
    
    # Remark: 'predict' returns a value from {-1, 1} and y is a binary value 
    n = len(X)
    expr = [0.5 / self.error, 0.5 / (1 - self.error)]
    for index in range(0, n):
      weights[index] *= expr[int(int(self.predict(X[index]) > 0) == y[index])]
  
  # Combine the prediction of this stump with its vote weight
  def vote(self, sample) -> float:
    return self.alpha * self.predict(sample)

@dataclass
class AdaBoost:
  learners: List[Stump]
  T: int = 100          # Number of rounds to run

  # The constructor
  def __init__(self, T):
    if T <= 0:
      print("AdaBoost needs a positive number of iterations")
      sys.exit(1)
    self.T = T
  
  # Check the data and transform it into a more convenient way
  def preprocessing(self, X, y):
    # Shrink y and sort the values, since we assume that the lower value represents the minus class
    shrinked = sorted(list(set(y)))
    if len(shrinked) == 1:
      print("Your training data is redundant. All samples fall into the same class")
      sys.exit(1)
    if len(shrinked) > 2:
      print("AdaBoost can support by now only binary classification")
      sys.exit(1)
    
    # Translate y into binary classes (-1, 1)
    n = len(y)
    z = []
    for index in range(0, n):
      z.append(int(y[index] == shrinked[1]))
  
    # Build up the container
    numFeatures = len(X[0])
    store = []
    for index in range(0, numFeatures):
      store.append(dict())
      
    # Each feature has a dictionary of values and each of these of values have a minusList and a plusList
    # minusList = list of positions 'pos' for which y[pos] = "-" (-1)
    # plusList = list of positions 'pos' for which y[pos] = "+" (+1)
    n = len(X)
    for indexInData in range(0, n):
      x = X[indexInData]
      for indexInSample in range(0, numFeatures):
        if not (x[indexInSample] in store[indexInSample]):
          store[indexInSample][x[indexInSample]] = [[indexInData], []] if not z[indexInData] else [[], [indexInData]]
        else:
          store[indexInSample][x[indexInSample]][z[indexInData]].append(indexInData)
        
    # Identify which type of variables we are dealing with
    types = []
    for index in range(0, numFeatures):
      size = len(store[index])
      if size == 1:
        print("One feature is redundant. Please take a look at your data again!")
        sys.exit(1)
      # Continous variable?
      if size >= 0.2 * n:
        types.append("continous")
      elif size == 2:
        types.append("binary")
      else:
        types.append("discrete")

    print("Data types: " + str(types))
        
    # Get rid of values, which do not affect the accuracy (this mainly applies for continous variables)
    for index in range(0, numFeatures):
      if types[index] == "continous":
        size = len(store[index])
        tmp = sorted(store[index].items())
        
        # Take the average between every continous value
        # For the last variable, only upshift it by 1
        save = []
        acc = [[], []]
        ptr = 0
        for (value, [minusList, plusList]) in tmp:
          acc[0] += minusList
          acc[1] += plusList
          # The last point in the list of values?
          if ptr == size - 1:
            # Add the last mid-point, which is 1 greater than the last value
            save.append((value + 1, [acc[0], acc[1]]))
          else:
            # Check up the parameters of the next value
            nextValue = tmp[ptr + 1][0]
            nextMinusList = tmp[ptr + 1][1][0]
            nextPlusList = tmp[ptr + 1][1][1]
            
            # Is this mid-point not redundant?
            if not ((not minusList and not nextMinusList) or (not plusList and not nextPlusList)):
              # Add the mid-point
              save.append(((value + nextValue) / 2, [acc[0], acc[1]]))
              
              # And clear the accumulated lists
              acc[0] = []
              acc[1] = []
          ptr += 1
        # And restore
        store[index] = save
      else:
        # For any other data type, keep it as it is now
        store[index] = sorted(store[index].items())
    # Return the modified data and the container
    return X, z, store
  
  # The fitter
  def fit(self, X, y) -> None:
    # Initialize the weights
    n = len(X)
    weights = np.array([1.0 / n] * n)

    numFeatures = len(X[0])
    X, y, store = self.preprocessing(X, y)

    # We use -1 and 0 interchangeably, do not get confused by that
    binClass = [[], []]
    for index in range(0, n):
      binClass[y[index]].append(index)

    # And compute
    self.learners = []
    for index in range(0, self.T):
      tmp = weights[binClass[0]].sum()
      totalWeights = [tmp, 1 - tmp]
      learner = Stump()
      for index in range(0, numFeatures):
        partialSum = [0, 0]
        
        # Note that the last iteration is in vain
        # Why? It simply tells us if the entire column is either minus or plus
        # If we wanted to get rid of it, in preprocessing we could remove the feature which has only one value
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
      
      # Compute alpha of learner and update weights
      if learner.getError():
        learner.updateWeights(X, y, weights)
      self.learners.append(learner)
    pass

  # Receives a query and outputs the hypothesis
  def query(self, sample):
    H = 0
    for learner in self.learners:
      H += learner.vote(sample)
    return int(H > 0)
  
  # Compute test error
  def score(self, X, y):
    if len(X) != len(y):
      print("x-axis and y-axis of test data do not have the same size")
      sys.exit(1)
    n = len(X)
    if not n:
      return 1
    else:
      # First define the binary class
      shrinked = sorted(list(set(y)))
      if len(shrinked) == 1:
        print("Your test data is redundant. All samples fall into the same class")
        return 1
      elif len(shrinked) > 2:
        print("AdaBoost can support by now only binary classification")
        sys.exit(1)
      
      # And compute the score
      correct = 0
      for index in range(0, n):
        # Note that 'shrinked' has already been sorted
        expected = int(y[index] == shrinked[1])
        predicted = self.query(X[index])
        correct += int(expected == predicted)
      accuracy = float(correct / n)
      return accuracy

def benchmark(dataName, tuning):
  dataReader = DataReader()
  X, y = dataReader.read(dataName)
  train_X, test_X, train_y, test_y = train_test_split(X, y)

  print("Train data: " + str(len(train_X)))
  print("Test data: " + str(len(test_X)))

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
