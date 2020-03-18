import os
import sys
import numpy as np
import math

# The dataclass type and its features
from dataclasses import dataclass
from typing import List
    
# A node in the decision tree
class Node:
  def __init__(self):
    self.coefficient = 1
    self.branches = None
    pass
  
  # Print a pretty decision tree
  def custom(self, tab):
    return self.to_string() + " branches=[" + ("]\n" if not self.branches else "\n" + "".join("\t" * tab + branch.custom(tab + 1) for branch in self.branches) + ("\t" * (tab - 1) + "]\n"))
  
  def __repr__(self) -> str:
    return self.custom(1)
    
  def to_string(self) -> str:
    return "Final node: result=" + str(self.result) if not self.branches else "Inner node: coefficient=" + str(self.coefficient) + " feature=" + str(self.feature) + " threshold=" + str(self.threshold)
  
  def setParameters(self, coefficient, feature, threshold):
    self.coefficient = coefficient
    self.feature = feature
    self.threshold = threshold
    
  def setBranches(self, branches):
    self.branches = branches
  
  # Actual computation of log(odds)
  def computeLogOdds(self, negativeCount, positiveCount) -> float:
    if not negativeCount:
      return math.inf
    elif not positiveCount:
      return -math.inf
    else:
      return math.log(float(positiveCount) / (negativeCount + positiveCount))
    
  # Compute the probability from log(odds)
  def computeProbability(self) -> float:
    val =  1.0 / (1.0 + math.exp(-self.result))
    return val

  # Save the log(odds) of the list
  def setResult(self, y, indexList) -> None:
    if not indexList:
      self.result = 0
      pass
    count = [0, 0]
    for index in indexList:
      count[y[index]] += 1
    self.result = self.computeLogOdds(count[0], count[1])
    pass

  # Find the node the samplesfalls into
  def traverse(self, sample):
    if not self.branches:
      return self
    nextBranch = self.branches[int(sample[self.feature] > self.threshold)]
    return nextBranch.traverse(sample)

# The decision tree
class DecisionTree:
  def __init__(self, coefficientName = "gini"):
    self.coefficientName = coefficientName
    pass
  
  def __repr__(self) -> str:
    return str(self.tree)
  
  def preprocessing(self, X, y):
    # Shrink y and sort the values, since we assume that the lower value represents the minus class
    shrinked = sorted(list(set(y)))
    if len(shrinked) == 1:
      print("Your training data is redundant. All samples fall into the same class")
      sys.exit(1)
    if len(shrinked) > 2:
      print("Decision tree can support by now only binary classification")
      sys.exit(1)
    
    # Translate y into binary classes (-1, 1)
    z = []
    for index in range(self.size):
      z.append(int(y[index] == shrinked[1]))
    y = z
  
    # Build up the container
    self.store = []
    for index in range(self.numFeatures):
      self.store.append(dict())
      
    # Each feature has a dictionary of values and each of these of values have a minusList and a plusList
    # minusList = list of positions 'pos' for which y[pos] = "-" (-1)
    # plusList = list of positions 'pos' for which y[pos] = "+" (+1)
    for indexInData in range(self.size):
      x = X[indexInData]
      for feature in range(self.numFeatures):
        if not (x[feature] in self.store[feature]):
          self.store[feature][x[feature]] = [[indexInData], []] if not y[indexInData] else [[], [indexInData]]
        else:
          self.store[feature][x[feature]][y[indexInData]].append(indexInData)
      
    # Identify which type of variables we are dealing with
    self.types = []
    for index in range(self.numFeatures):
      size = len(self.store[index])
      if size == 1:
        print("One feature is redundant. Please take a look at your data again!")
        sys.exit(1)
      # Continous variable?
      if size >= 0.2 * self.size:
        self.types.append("continous")
      elif size == 2:
        self.types.append("binary")
      else:
        self.types.append("discrete")
    
    # Compress the list of possible values for discrete/continous features
    for feature in range(self.numFeatures):
      if self.types[feature] != "binary":
        tmp = sorted(self.store[feature].items())
        size = len(tmp)
        save = []
        for ptr, (value, [minusList, plusList]) in enumerate(tmp):
          if ptr == size - 1:
            save.append(value + 1)
          else:
            nextValue = tmp[ptr + 1][0]
            nextMinusList = tmp[ptr + 1][1][0]
            nextPlusList = tmp[ptr + 1][1][1]
            if not ((not minusList and not nextMinusList) or (not plusList and not nextPlusList)):
              save.append((value + nextValue) / 2)
        self.store[feature] = save
      else:
        self.store[feature] = sorted(self.store[feature].keys())
    return y
  
  # The information entropy
  def f(self, x):
    if not x:
      return 0
    return -x * math.log(x, 2)

  # Compute the entropy of the current split, which has generated the 2x2 matrix 'counts'
  def computeEntropy(self, counts):
    rev = 1.0 / counts.sum()
    alfa = counts[0][0]
    beta = counts[0][1]
    gamma = counts[1][0]
    delta = counts[1][1]
    return rev * (self.f(alfa) + self.f(beta) + self.f(gamma) + self.f(delta) - self.f(alfa + beta) - self.f(gamma + delta))
  
  # Compute the Gini index of the current split, which has generated the 2x2 matrix 'counts'
  def computeGiniIndex(self, counts):
    rev = 1.0 / counts.sum()
    alfa = counts[0][0]
    beta = counts[0][1]
    gamma = counts[1][0]
    delta = counts[1][1]
    return 1 - rev * ((alfa**2 + beta**2) / (alfa + beta) + (gamma**2 + delta**2) / (gamma + delta))
  
  # Compute the coefficient (either Gini index or the information entropy)
  def computeCoefficient(self, counts):
    return self.computeGiniIndex(counts) if self.coefficientName == "gini" else self.computeEntropy(counts)
  
  # Build up the tree
  def process(self, X, y, indexList, featureList) -> Node:
    currNode = Node()
    
    # Can we stop?
    if len(indexList) <= self.nodeSize:
      currNode.setResult(y, indexList)
      return currNode
    
    # Iterate through all features and take the smallest coefficent of the split
    for feature in featureList:
      for value in self.store[feature]:
        counts = np.array([[0, 0], [0, 0]])
        for index in indexList:
          counts[int(X[index][feature] > value)][y[index]] += 1
        # Is this split destructive?
        if not counts[0].sum() or not counts[1].sum():
          continue
        # And compute the coefficent
        coefficient = self.computeCoefficient(counts)
        if coefficient < currNode.coefficient:
          currNode.setParameters(coefficient, feature, value)
    # No good split?
    if currNode.coefficient == 1:
      currNode.setResult(y, indexList)
      return currNode
    
    # Split up the list of indexes
    indexes = [[], []]
    for index in indexList:
      indexes[int(X[index][currNode.feature] > currNode.threshold)].append(index)
    
    # And recursively build up the branches
    # If the current split was done upon a binary feature, we eliminate for the upcoming branches
    newFeatureList = list(filter(lambda feature: feature != currNode.feature, featureList)) if self.types[currNode.feature] == "binary" else featureList
    currNode.setBranches([self.process(X, y, indexes[0], newFeatureList), self.process(X, y, indexes[1], newFeatureList)])
    return currNode
  
  # The fitter
  def fit(self, X, y) -> None:
    self.size = len(X)
    self.nodeSize = math.log(self.size, 2)
    self.numFeatures = len(X[0])
    y = self.preprocessing(X, y)
    self.tree = self.process(X, y, range(self.size), range(self.numFeatures))
    
  # The predictor 
  def predict(self, sample):
    result = self.tree.traverse(sample)
    return int(result.computeProbability() > 0.5)
   
  # Compute the accuracy
  def score(self, X, y):
    if len(X) != len(y):
      print("x-axis and y-axis of test data do not have the same size")
      sys.exit(1)
    n = len(X)
    if not n:
      return 1
    # First define the binary class
    shrinked = sorted(list(set(y)))
    if len(shrinked) == 1:
      print("Your test data is redundant. All samples fall into the same class")
      return 1
    elif len(shrinked) > 2:
      print("Decision trees can support by now only binary classification")
      sys.exit(1)
     
    # And compute the score
    correctPredicted = 0
    n = len(X)
    for index in range(n):
      expected = int(y[index] == shrinked[1])
      predicted = self.predict(X[index])
      correctPredicted += int(predicted == expected)
    return float(correctPredicted / n)
