import os
import sys
import numpy as np
import math

# The dataclass type and its features
from dataclasses import dataclass
from typing import List
    
class Node:
  def __init__(self):
    self.gain = 1
    pass
  
  def setParameters(self, gain, feature, threshold = 0):
    self.gain = gain
    self.feature = feature
    self.threshold = threshold
    
  def setBranches(self, branches):
    self.branches = branches
    
@dataclass
class DecisionTree:
  tree: Node
  size: int = 0
  numFeatures: int = 0
  
  def __init__(self):
    pass
  
  def formula(x):
    return -x * log(x, 2))
  
  def computeEntropy(x, y):
    total = 1.0 / (x + y)
    return formula(x * total, y * total)
  
  def process(self, X, y, indexList, currTree = None) -> Node:
    if len(indexList) < 20:
      return None
    currTree = Node()
    for feature in range(self.numFeatures):
      # We differentiate between continous and non-continous variables
      # Why? Because in the case of a continous one, we should choose the best threshold
      if types[feature] == "continous":
        for continousValue in store[feature]:
          count = np.array([[0, 0], [0, 0])
          for index in indexList:
            count[int(X[index][feature] <= continousValue)][y[index]] += 1
          totalCount = 1.0 / count.sum()
          entropySum = totalCount * (count[0].sum() computeEntropy(count[0][0], count[0][1]) + count[1].sum() * computeEntropy(count[1][0], count[1][1]))
          if entropySum < currTree.gain:
            currTree.setParameters(entropySum, feature, threshold)
      else:
        totalCount = 0
        entropySum = 0
        for discreteValue in store[feature]:
          count = [0, 0]
          for index in indexList:
            if X[index][feature] == discreteValue:
              count[y[indexList]] += 1
          totalCount += count[0] + count[1]
          entropySum += computeEntropy(count[0], count[1])
        entropySum /= totalCount
        if entropySum < currTree.gain:
          currTree.setParameters(entropySum, feature)
    if types[currTree.feature] == "continous":
      indexes = [[], []]
      for index in indexList:
        indexes[int(X[index][currTree.feature] <= currTree.threshold)].append(index)
      branches = [self.process(X, y, indexes[0]), self.process(X, y, indexes[1])]
      currTree.setBranches(branches)
    else:
      span = len(store[currTree.feature])
      indexes = []
      for i in range(span):
        indexes.append(list())
      for index in indexList:
        indexes[mapValue[X[index][currTree.feature]]].append(index)
      branches = []
      for pos in range(span):
        branches.append(self.process(X, y, indexes[pos]))
      currTree.setBranches(branches)
    return currTree
    
  def fit(self, X, y) -> None:
    self.size = len(X)
    self.numFeatures = len(X[0])
    self.tree = self.process(X, y, range(n))
   
  def score(self, X, y) -> float:
    return 1.0
