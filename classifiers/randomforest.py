import random
import numpy
import math

from matplotlib import pyplot as plt
from classifiers.baseline import Classifier
from classifiers.decisiontree import DecisionTree

class RandomForest(Classifier):
  def __init__(self):
    self.maxNumTrees = 50
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
    y = self.preprocessing(X, y)
    self.trees = []
    constRange = range(self.size)
    for treeIndex in range(self.maxNumTrees):
      # Take a list of indexes, with replacement
      currList = random.choices(constRange, k=self.size)
      
      # Fit the selected data with a reduced number of features
      currTree = self.fitDT(X, y, currList, int(math.sqrt(self.numFeatures)))
      
      # And add the new tree
      self.trees.append(currTree)
    pass
  
  # Predict by adding up the probabilities of each leaf node in which samples falls into
  def predict(self, sample):
    vote = 0
    for tree in self.trees:
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
