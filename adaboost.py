import os
import sys
import numpy as np
import math
import time

# The dataclass type and its features
from dataclasses import dataclass
from typing import List

# The binary search for the performant prediction
from bisect import bisect_left

####
# A performant implementation with auto-tuning of AdaBoost ensemble learning algorithm
####

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

  # Compute alpha which reduces the error bound of the total error
  def computeAlpha(self) -> None:
    if not (self.error in [0, 1]):
      self.alpha = 0.5 * math.log((1 - self.error) / self.error)
  
  # Return the prediction of this stump
  def predict(self, sample) -> int:
    return self.PERM[self.side][int(sample[self.feature] <= self.threshold)]
  
  # Rescale the weights, as described in MIT youtube video (https://www.youtube.com/watch?v=UHBmv7qCey4)
  def updateWeights(self, X, y, weights, totalMinusWeight) -> None:
    # First compute alpha
    self.computeAlpha()
    
    # Remark: 'predict' returns a value from {-1, 1} and y is a binary value 
    # Also update the sum of weights of those samples in the minus class
    n = len(X)
    totalMinusWeight[0] = 0
    expr = [0.5 / self.error, 0.5 / (1 - self.error)]
    for index in range(n):
      weights[index] *= expr[int(int(self.predict(X[index]) > 0) == y[index])]
      totalMinusWeight[0] += weights[index] if not y[index] else 0
    pass
  
  # Combine the prediction of this stump with its vote weight
  def vote(self, sample) -> float:
    return self.alpha * self.predict(sample)

@dataclass
class AdaBoost:
  learners: List[Stump]
  numFeatures: int = 0
  DEFAULT_T = 5
  MAX_EFFORT = 1e5

  # The constructor
  def __init__(self):
    pass
  
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
    for index in range(n):
      z.append(int(y[index] == shrinked[1]))
    y = z
  
    # Build up the container
    store = []
    self.numFeatures = len(X[0])
    for index in range(self.numFeatures):
      store.append(dict())
      
    # Each feature has a dictionary of values and each of these of values have a minusList and a plusList
    # minusList = list of positions 'pos' for which y[pos] = "-" (-1)
    # plusList = list of positions 'pos' for which y[pos] = "+" (+1)
    for indexInData in range(n):
      x = X[indexInData]
      for indexInSample in range(self.numFeatures):
        if not (x[indexInSample] in store[indexInSample]):
          store[indexInSample][x[indexInSample]] = [[indexInData], []] if not y[indexInData] else [[], [indexInData]]
        else:
          store[indexInSample][x[indexInSample]][y[indexInData]].append(indexInData)
        
    # Identify which type of variables we are dealing with
    types = []
    for index in range(self.numFeatures):
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
    for index in range(self.numFeatures):
      if types[index] == "continous":
        size = len(store[index])
        tmp = sorted(store[index].items())
        
        # Take the average between every continous value
        # For the last variable, only upshift it by 1
        save = []
        acc = [[], []]
        for ptr, (value, [minusList, plusList]) in enumerate(tmp):
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
        # And restore
        store[index] = save
      else:
        # For any other data type, keep it as it is now
        store[index] = sorted(store[index].items())
    # Return the modified data and the container
    return y, store
  
  # The fitter
  def fit(self, X, y) -> None:
    # Initialize the weights
    n = len(X)
    weights = np.array([1.0 / n] * n)

    numFeatures = len(X[0])
    y, store = self.preprocessing(X, y)

    # Compute the initial weights of the "-" class
    totalMinusWeight = [(1.0 / n) * len(list(filter(lambda index: not y[index], range(n))))]
      
    # Compute the number of rounds to run
    T = self.DEFAULT_T
    if n < self.MAX_EFFORT:
      T = int(self.MAX_EFFORT / n)
      
    # And compute
    self.learners = []
    for iteration in range(T):
      learner = Stump()
      for index in range(self.numFeatures):
        # Note that the last iteration is in vain
        # Why? It simply tells us if the entire column is either minus or plus
        # If we wanted to get rid of it, in preprocessing we could remove the feature which has only one value
        partialSumOfDeltas = 0
        for (discreteValue, [minusList, plusList]) in store[index]:
          partialSumOfDeltas += weights[plusList].sum() - weights[minusList].sum()
        
          # Compute the sum of weights of the mispredicted samples.
          # We can compute only the error when, for the current feature,
          # the samples with a less or equal value receive a 'minus' classification,
          # since the other case around is symmetric.
          minusError = partialSumOfDeltas + totalMinusWeight[0]
          plusError = 1 - minusError
          
          # And update the learner (only one error could influence the current learner)
          if minusError < min(plusError, learner.error):
            learner.setParameters(minusError, index, discreteValue, 0)
          elif plusError < min(minusError, learner.error):
            learner.setParameters(plusError, index, discreteValue, 1)
      
      # Compute alpha of learner and update weights
      if learner.error:
        learner.updateWeights(X, y, weights, totalMinusWeight)
      self.learners.append(learner)
    pass

  # Receives a query and outputs the hypothesis
  def query(self, sample):
    H = 0
    for learner in self.learners:
      H += learner.vote(sample)
    return int(H > 0)
  
  # Compute test error
  def score(self, X, y) -> float:
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
      
      # Put the learners with the same feature into the same bucket
      commonFeature = []
      for feature in range(self.numFeatures):
        commonFeature.append(list())
      for learner in self.learners:
        commonFeature[learner.feature].append(learner)
      
      # Take only those features, the buckets of which are not empty
      mapFeature = []
      for feature in range(self.numFeatures):
        if commonFeature[feature]:
          mapFeature.append(feature)
          
      # And get rid of those empty buckets
      commonFeature = list(filter(lambda elem: elem, commonFeature))
      
      # We preprocess the sum votes from each classifier, by sorting the thresholds (which should be unique)
      # The first sum 'sumGreaterOrEqualThanSample' sums up the votes of those classifiers, the thresholds of which
      # are greater of equal than the value of the current sample
      # The second sum, in the same way, but note that the construction differs
      # In order to cover the case, in which a sample comes and its value is strictly greater than all thresholds
      # of the respective feature (in which case the binary search will return as index the size of learners),
      # we insert a sentinel at the end of 'votes' for each feature.
      prepared = []
      onlyThresholds = []
      for index, bucketList in enumerate(commonFeature):
        sortedBucketList = sorted(bucketList, key=lambda learner: learner.threshold)
        
        # Build up the partial sum 'sumGreaterOrEqualThanSample'
        # Note that we start from the beginning, since all elements which are on our right have a greater or equal threshold
        votes = []
        sumGreaterOrEqualThanSample = 0
        for ptr, learner in enumerate(sortedBucketList):
          votes.append(sumGreaterOrEqualThanSample)
          sumGreaterOrEqualThanSample += learner.alpha * learner.PERM[learner.side][int(False)]
        # And add the sentinel: gather all votes, when the value of the sample is simply strictly greater than all thresholds of this feature
        votes.append(sumGreaterOrEqualThanSample)
       
        # Build up the partial sum 'sumLowerThanSample'
        # Note that we start from the end, since all elements which are on our left have a threshold strictly lower than the threshold of the current learner
        sumLowerThanSample = 0
        ptr = len(sortedBucketList)
        while ptr != 0:
          learner = sortedBucketList[ptr - 1]
          sumLowerThanSample += learner.alpha * learner.PERM[learner.side][int(True)]
          
          # And add it to the already computed partial sum in 'votes'
          votes[ptr - 1] += sumLowerThanSample
          ptr -= 1
        # Add the votes of this feature and keep only the thresholds from each learner
        prepared.append(votes)
        onlyThresholds.append(list(map(lambda learner: learner.threshold, sortedBucketList)))
        
      # And compute the score
      correctClassified = 0
      for index in range(n):
        # Note that 'shrinked' has already been sorted
        expected = int(y[index] == shrinked[1])
        
        # This is an improved way to compute the prediction
        # If for any reasons, you want to go the safe way,
        # you can use the function 'self.query(X[index])', which computes it in the classical way
        predicted = 0
        for notNullFeature, votes in enumerate(prepared):
          # Note that 'votes' has a sentinel at the end to capture the case where the value of sample is strictly greater than all thresholds of 'mapFeature[notNullFeature]'
          pos = bisect_left(onlyThresholds[notNullFeature], X[index][mapFeature[notNullFeature]])
          predicted += votes[pos]
        predicted = int(predicted > 0)
        correctClassified += int(expected == predicted)
      accuracy = float(correctClassified / n)
      return accuracy
