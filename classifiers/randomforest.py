import random
import numpy as np
import math

from matplotlib import pyplot as plt
from classifiers.baseline import Classifier
from classifiers.decisiontree import DecisionTree

class RandomForests(Classifier):
  def __init__(self, maxNumTrees = 100):
    self.maxNumTrees = maxNumTrees
    pass
  
  class RandomDecisionTree(DecisionTree):
    # The predictor 
    def predict(self, sample):
      result = self.tree.traverse(sample)
      return result.computeProbability()
    
  # Fit the current decision tree
  def fitDT(self, X, y, indexList, randomSize):
    tree = self.RandomDecisionTree()
    newX = []
    newY = []
    for index in indexList:
      newX.append(X[index])
      newY.append(y[index])
    tree.fit(newX, newY, randomSize, self.types, self.store)
    return tree
  
  def fit(self, X, y):
    if True:
      y = self.preprocessing(X, y)
      self.all = []
      for randomSize in range(1, self.numFeatures + 1):
        print("try now with " + str(randomSize))
        self.trees = []
        constRange = range(self.size)
        confusionMatrix = np.array([[0, 0], [0, 0]])
        for treeIndex in range(self.maxNumTrees):
          # Take a list of indexes, with replacement
          currList = random.choices(constRange, k=self.size)
          
          # Fit the selected data with a reduced number of features
          currTree = self.fitDT(X, y, currList, randomSize)
          
          # And add the new tree
          self.trees.append(currTree)
        self.all.append(self.trees)
    else:
      y = self.preprocessing(X, y)
      self.trees = []
      constRange = range(self.size)
      randomSize = int(math.log(self.numFeatures, 2)) + 1
      for treeIndex in range(self.maxNumTrees):
        # Take a list of indexes, with replacement
        currList = random.choices(constRange, k=self.size)
        
        # Fit the selected data with a reduced number of features
        currTree = self.fitDT(X, y, currList, randomSize)
        
        # And add the new tree
        self.trees.append(currTree)
    pass
  
  def checkFitting(self, X, y):
    if True:
      self.checkTestData(X, y)
      acc = []
      for randomSize in range(1, self.numFeatures + 1):
        print("check now : " + str(randomSize))
        scores = []
        for treeIndex in range(1, self.maxNumTrees + 1):
          correctPredicted = 0
          for index, sample in enumerate(X):
            expected = int(y[index] == self.shrinked[1])
            correctPredicted += int(self.predict(sample, randomSize - 1, treeIndex) == expected)
          scores.append(float(correctPredicted / len(X)))
        print(scores)
        acc.append(scores)
      return acc
    else:
      self.checkTestData(X, y)
      
      scores = []
      for treeIndex in range(1, self.maxNumTrees + 1):
        correctPredicted = 0
        for index, sample in enumerate(X):
          expected = int(y[index] == self.shrinked[1])
          correctPredicted += int(self.predict(sample, treeIndex) == expected)
        scores.append(float(correctPredicted / len(X)))
      return scores
  
  # Predict by adding up the probabilities of each leaf node in which samples falls into
  def predict(self, sample, ensembleIndex, treeIndex):
    vote = 0
    for tree in self.all[ensembleIndex][:treeIndex]:
      vote += tree.predict(sample)
    vote /= self.maxNumTrees
    return int(vote > 0.5)
    
  def score(self, X, y):
    # Check the test data
    self.checkTestData(X, y)

    # And compute the score
    correctPredicted = 0
    for index, sample in enumerate(X):
      expected = int(y[index] == self.shrinked[1])
      correctPredicted += int(self.predict(sample) == expected)
    return float(correctPredicted / len(X))
