import baseline
import numpy
import math
    
# The decision tree
class DecisionTree(baseline.Classifier):  
  def __init__(self, coefficientName = "gini"):
    self.classifierName = "Decision Tree"
    self.coefficientName = coefficientName
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
  def process(self, X, y, indexList, featureList) -> Node:
    currNode = self.Node()
    
    # Can we stop?
    if len(indexList) <= self.nodeSize:
      currNode.setResult(y, indexList)
      return currNode
    
    # Iterate through all features and take the smallest coefficent of the split
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
    # Check the test data
    self.checkTestData(X, y)
    
    # And compute the score
    correctPredicted = 0
    n = len(X)
    for index in range(n):
      expected = int(y[index] == self.shrinked[1])
      predicted = self.predict(X[index])
      correctPredicted += int(predicted == expected)
    return float(correctPredicted / n)
