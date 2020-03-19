from classifiers.baseline import Classifier
import numpy
import math
import random    

# The decision tree
class DecisionTree(Classifier):  
  def __init__(self, coefficientName = "gini", maxDepth = -1, master = None):
    self.coefficientName = coefficientName
    self.maxDepth = maxDepth
    self.master = master
    pass
  
  def __repr__(self) -> str:
    return str(self.tree)

  ####
  # The node in the decision tree
  ####
  
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
        return "Final node: count/agg=" + str(self.count) if not self.branches else "Inner node: coefficient=" + str(self.coefficient) + " feature=" + str(self.feature) + " threshold=" + str(self.threshold)
    
    def setParameters(self, coefficient, feature, threshold):
      self.coefficient = coefficient
      self.feature = feature
      self.threshold = threshold
      
    def setBranches(self, branches):
      self.branches = branches
  
    # Save the log(odds) of the list
    def setResult(self, y, indexList, master = None) -> None:
      # In case we only compute the probability
      if not master:
        self.count = [0, 0]
        for index in indexList:
          self.count[y[index]] += 1
      else:
        # In case we use Gradient Boost:
        # We received a master, which contains the list of probabilites
        # At this step, we compute the output of the current leaf node
        # Moreover, we update the probabilites, based on the previous sum of odds and the current one
        self.count = 0
        residualSum = 0
        probabilitySum = 0
        for index in indexList:
          residualSum += y[index] - master.probabilites[index][1]
          probabilitySum += master.probabilites[index][1] * (1 - master.probabilites[index][1])
        self.count = residualSum / probabilitySum
        for index in indexList:
          newOdds = master.probabilites[index][0] + master.learningRate * self.count
          master.probabilites[index][1] = 1.0 / (1 + math.exp(-newOdds))
        
    # Compute the probability of this node
    def computeProbability(self):
      if not self.count[0]:
        return 1
      elif not self.count[1]:
        return 0
      return self.count[1] / (self.count[0] + self.count[1])

    # Find the node the samples falls into
    def traverse(self, sample):
      if not self.branches:
        return self
      nextBranch = self.branches[int(sample[self.feature] > self.threshold)]
      return nextBranch.traverse(sample)
  
  ####
  # End of the node in the decision tree
  #### 
  
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
  def process(self, X, y, indexList, featureList, currentDepth) -> Node:
    currNode = self.Node()
    
    # Can we stop?
    if (not len(featureList)) or (len(indexList) <= self.nodeSize):
      currNode.setResult(y, indexList, self.master)
      return currNode
    # Are we limited by a maximum depth?
    if (self.maxDepth != -1) and (currentDepth > self.maxDepth):
      currNode.setResult(y, indexList, self.master)
      return currNode
    
    # Iterate through all features and take the smallest coefficent of the split
    if self.randomSize != 0 and self.randomSize < len(featureList):
      featureList = random.sample(featureList, self.randomSize) 
    for feature in featureList:
      for value in self.store[feature]:
        counts = numpy.array([[0, 0], [0, 0]])
        for index in indexList:
          counts[int(X[index][feature] > value)][y[index]] += 1
        # Is this split destructive?
        if not counts[0].sum() or not counts[1].sum():
          continue
        # And compute the coefficent
        coefficient = self.computeCoefficient(counts)
        if coefficient < currNode.coefficient:
          currNode.setParameters(coefficient, feature, value)
    # No good split found?
    if currNode.coefficient == 1:
      currNode.setResult(y, indexList, self.master)
      return currNode
    
    # Split up the list of indexes
    indexes = [[], []]
    for index in indexList:
      indexes[int(X[index][currNode.feature] > currNode.threshold)].append(index)
    
    # And recursively build up the branches
    # If the current split was done upon a binary feature, we eliminate for the upcoming branches
    newFeatureList = list(filter(lambda feature: feature != currNode.feature, featureList)) if self.types[currNode.feature] == "binary" else featureList
    currNode.setBranches([self.process(X, y, indexes[0], newFeatureList, currentDepth + 1), self.process(X, y, indexes[1], newFeatureList, currentDepth + 1)])
    return currNode
  
  # The fitter
  def fit(self, X, y, randomSize = 0, types = None, store = None) -> None:
    self.randomSize = randomSize
    if not randomSize:
      y = self.preprocessing(X, y)
    else:
      self.size = len(X)
      self.numFeatures = len(X[0])
      self.types = types
      self.store = store
    self.nodeSize = math.log(self.size, 2)
    self.tree = self.process(X, y, range(self.size), range(self.numFeatures), 0)
   
  # The predictor 
  def predict(self, sample):
    result = self.tree.traverse(sample)
    return int(result.computeProbability() > 0.5)
  
  # Compute the accuracy
  def score(self, X, y):
    # Check the test data
    self.checkTestData(X, y)
    
    # And compute the score
    correctPredicted = 0
    for index, sample in enumerate(X):
      expected = int(y[index] == self.shrinked[1])
      predicted = self.predict(sample)
      correctPredicted += int(predicted == expected)
    return float(correctPredicted / len(X))
