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
  
  def custom(self, tab):
    return self.to_string() + " branches=[" + ("]\n" if not list(filter(lambda branch: branch != None, self.branches)) else ("\n" + "".join("\t" * tab + (branch.custom(tab + 1) if branch != None else "None" + ("" if index == len(self.branches) - 1 else "\n")) for index, branch in enumerate(self.branches)) + ("\n" + "\t" * (tab - 1) + "]\n"))) if self != None else ""
  
  def __repr__(self) -> str:
    return self.custom(1)
  
  def to_string(self) -> str:
    return "Node: gain=" + str(self.gain) + " feature=" + str(self.feature) + " type=" + str(self.featureType) + " & values=" + (str(self.threshold) if self.featureType == "continous" else str(self.discreteMap))
  
  def setParameters(self, gain, types, feature, thresholdOrDiscreteMap):
    self.gain = gain
    self.feature = feature
    self.featureType = types[feature]
    if self.featureType == "continous":
      self.threshold = thresholdOrDiscreteMap
    else:
      self.discreteMap = thresholdOrDiscreteMap
    
  def setBranches(self, branches):
    self.branches = branches
  
  def traverse(self, sample, acc = ""):
    if self == None:
      return acc
    nextBranch = self.branches[int(sample[self.feature] <= self.threshold)] if self.featureType == "continous" else self.branches[self.discreteMap.get(sample[self.feature])]
    acc += self.to_string()
    if nextBranch == None:
      return acc
    return nextBranch.traverse(sample, acc + "\n")

class DecisionTree:
  def __init__(self):
    pass
  
  def __repr__(self) -> str:
    return str(self.tree)
  
  def formula(self, x):
    return -x * math.log(x, 2)
  
  def computeEntropy(self, x, y):
    total = 1.0 / (x + y)
    if not x or not y:
      return 1
    return self.formula(x * total) + self.formula(y * total)
  
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
    z = []
    for index in range(self.size):
      z.append(int(y[index] == shrinked[1]))
  
    # Build up the container
    self.store = []
    self.mapValue = []
    for index in range(self.numFeatures):
      self.store.append(dict())
      self.mapValue.append(dict())
      
    for indexInData in range(self.size):
      x = X[indexInData]
      for feature in range(self.numFeatures):
        self.store[feature][x[feature]] = True
        
    for feature in range(self.numFeatures):
      tmp = sorted(list(self.store[feature].keys()))
      for index, value in enumerate(tmp):
        self.mapValue[feature][value] = index
      self.store[feature] = tmp
      
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

    print("Data types:")
    for index, type_ in enumerate(self.types):
      print(str(index) + " -> " + str(type_) + " values: " + str(self.store[index]))
    return z
  
  def process(self, X, y, indexList, featureList) -> Node:
    if len(indexList) < 20:
      return None
    currNode = Node()
    # print("Now with range=" + str(indexList))
    # print("featureList: " + str(featureList))
    for feature in featureList:
      # We differentiate between continous and non-continous variables
      # Why? Because in the case of a continous one, we should choose the best threshold
      # print("try feature: " + str(feature))
      if self.types[feature] == "continous":
        for continousValue in self.store[feature]:
          count = np.array([[0, 0], [0, 0]])
          for index in indexList:
            count[int(X[index][feature] <= continousValue)][y[index]] += 1
          totalCount = 1.0 / count.sum()
          entropySum = totalCount * (count[0].sum() * self.computeEntropy(count[0][0], count[0][1]) + count[1].sum() * self.computeEntropy(count[1][0], count[1][1]))
          if entropySum < currNode.gain:
            currNode.setParameters(entropySum, self.types, feature, threshold)
      else:
        totalCount = 0
        entropySum = 0
        for discreteValue in self.store[feature]:
          count = [0, 0]
          for index in indexList:
            if X[index][feature] == discreteValue:
              count[y[index]] += 1
          totalCount += count[0] + count[1]
          entropySum += self.computeEntropy(count[0], count[1])
        entropySum /= totalCount
        if entropySum < currNode.gain:
          currNode.setParameters(entropySum, self.types, feature, self.mapValue[feature])
    if self.types[currNode.feature] == "continous":
      indexes = [[], []]
      for index in indexList:
        indexes[int(X[index][currNode.feature] <= currNode.threshold)].append(index)
      branches = [self.process(X, y, indexes[0], featureList), self.process(X, y, indexes[1], featureList)]
      currNode.setBranches(branches)
    else:
      nextFeatureList = list(filter(lambda feature: feature != currNode.feature, featureList))
      span = len(self.store[currNode.feature])
      indexes = []
      for i in range(span):
        indexes.append(list())
      for index in indexList:
        indexes[self.mapValue[currNode.feature][X[index][currNode.feature]]].append(index)
      branches = []
      for pos in range(span):
        branches.append(self.process(X, y, indexes[pos], nextFeatureList))
      currNode.setBranches(branches)
    return currNode
    
  def fit(self, X, y) -> None:
    self.size = len(X)
    self.numFeatures = len(X[0])
    y = self.preprocessing(X, y)
    self.tree = self.process(X, y, range(self.size), range(self.numFeatures))
    print(self)
  
  def predict(self, sample):
    return "sample: " + str(sample) + " got:\n" + self.tree.traverse(sample)
   
  def score(self, X, y):
    print(self.predict(X[0]))
