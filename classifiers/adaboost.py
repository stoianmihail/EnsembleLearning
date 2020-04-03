from classifiers.baseline import Classifier
import numpy
import math

# The dataclass type and its features
from dataclasses import dataclass
from typing import List

# The binary search for the performant prediction
from bisect import bisect_left

####
# A performant implementation with auto-tuning of AdaBoost ensemble learning algorithm
####

class AdaBoost(Classifier):
  # The constructor
  def __init__(self, iterations = 0):
    self.MAX_T = iterations
    self.DEFAULT_NUM_LEARNERS = 5
    self.MAX_EFFORT = 1e5
    pass

  ####
  # A decision stump (one-level decision tree)
  ####

  @dataclass
  class Stump:
    error: float = 1     # Used for computing the minimum error 
    alpha: float = 1     # This stump can be the best one
    feature: float = 0
    threshold: float = 0
    side: int = 0        # binary value: 0 -> less or equal values are "-", 1 -> otherwise
    PERM = numpy.array([[1, -1], [-1, 1]])

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
        print(self.alpha)
        self.alpha = 0.5 * math.log((1 - self.error) / self.error)
      elif not self.error:
        self.alpha = math.inf
      else:
        self.alpha = -math.inf
      
    # Return the prediction of this stump
    def predict(self, sample) -> int:
      return self.PERM[self.side][int(sample[self.feature] <= self.threshold)]
    
    # Rescale the weights, as described in MIT youtube video (https://www.youtube.com/watch?v=UHBmv7qCey4)
    def updateWeights(self, X, y, weights, totalMinusWeight) -> None:
      # First compute alpha
      self.computeAlpha()
        
      # Remark: 'predict' returns a value from {-1, 1} and y is a binary value 
      # Also update the sum of weights of those samples in the minus class
      totalMinusWeight[0] = 0
      expr = [0.5 / self.error, 0.5 / (1 - self.error)]
      for index, sample in enumerate(X):
        weights[index] *= expr[int(int(self.predict(sample) > 0) == y[index])]
        totalMinusWeight[0] += weights[index] if not y[index] else 0
      pass
    
    # Combine the prediction of this stump with its vote weight
    def vote(self, sample) -> float:
      return self.alpha * self.predict(sample)
    
  ####
  # End of the decision stump (one-level decision tree)
  ####
    
  # The fitter
  def fit(self, X, y) -> None:
    # Preprocess the data
    y = self.preprocessing(X, y, "withLists")

    # Initialize the weights
    weights = numpy.array([1.0 / self.size] * self.size)

    # Compute the initial weights of the "-" class
    totalMinusWeight = [(1.0 / self.size) * len(list(filter(lambda index: not y[index], range(self.size))))]
      
    # Compute the number of rounds to run
    T = self.MAX_T
    if not T:
      T = self.DEFAULT_NUM_LEARNERS
      if self.size < self.MAX_EFFORT:
        T = int(self.MAX_EFFORT / self.size)
      
    # And compute
    self.learners = []
    for iteration in range(T):
      print("now: " + str(iteration))
      learner = self.Stump()
      for feature in range(self.numFeatures):
        # Note that the last iteration is in vain
        # Why? It simply tells us if the entire column is either minus or plus
        # If we wanted to get rid of it, in preprocessing we could remove the feature which has only one value
        partialSumOfDeltas = 0
        for (discreteValue, [minusList, plusList]) in self.store[feature]:
          partialSumOfDeltas += weights[plusList].sum() - weights[minusList].sum()
        
          # Compute the sum of weights of the mispredicted samples.
          # We can compute only the error when, for the current feature,
          # the samples with a less or equal value receive a 'minus' classification,
          # since the other case around is symmetric.
          minusError = partialSumOfDeltas + totalMinusWeight[0]
          plusError = 1 - minusError
          
          # And update the learner (only one error could influence the current learner)
          if minusError < min(plusError, learner.error):
            learner.setParameters(minusError, feature, discreteValue, 0)
          elif plusError < min(minusError, learner.error):
            learner.setParameters(plusError, feature, discreteValue, 1)
      
      # Compute alpha of learner and update weights
      if learner.error:
        learner.updateWeights(X, y, weights, totalMinusWeight)
      self.learners.append(learner)
    pass

  # Receives a query and outputs the hypothesis
  def predict(self, sample):
    H = 0
    for learner in self.learners:
      H += learner.vote(sample)
    return int(H > 0)
  
  # Compute test accuracy
  def score(self, X, y, size = -1) -> float:
    # Check the test data
    self.checkTestData(X, y)
    
    tmp = self.learners
    if size != -1:
      tmp = self.learners[:size]
    
    # Put the learners with the same feature into the same bucket
    commonFeature = []
    for feature in range(self.numFeatures):
      commonFeature.append(list())
    for learner in tmp:
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
    for index in range(self.size):
      # Note that 'shrinked' has already been sorted
      expected = int(y[index] == self.shrinked[1])
      
      # This is an improved way to compute the prediction
      # If for any reasons, you want to go the safe way,
      # you can use the function 'self.predict(X[index])', which computes it in the classical way
      predicted = 0
      for notNullFeature, votes in enumerate(prepared):
        # Note that 'votes' has a sentinel at the end to capture the case where the value of sample is strictly greater than all thresholds of 'mapFeature[notNullFeature]'
        pos = bisect_left(onlyThresholds[notNullFeature], X[index][mapFeature[notNullFeature]])
        predicted += votes[pos]
      predicted = int(predicted > 0)
      correctClassified += int(expected == predicted)
    accuracy = float(correctClassified / self.size)
    return accuracy
