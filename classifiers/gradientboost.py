import math

from classifiers.baseline import Classifier
from classifiers.decisiontree import DecisionTree

class GradientBoost(Classifier):
  def __init__(self):
    self.learningRate = 0.2
    self.maxNumTrees = 50
    self.treeDepth = 4
    
  # The constituent: a small decision tree, which is limited by 'treeDepth'
  class SmallDecisionTree(DecisionTree):
    def __init__(self, master):
      self.size = master.size
      self.numFeatures = master.numFeatures
      self.types = master.types
      self.store = master.store
      super().__init__(maxDepth = master.treeDepth, master = master)
    
    def fit(self, X, y):
      self.randomSize = 0
      self.nodeSize = math.log(self.size, 2)
      self.tree = self.process(X, y, range(self.size), range(self.numFeatures), 0)
    
    def predict(self, sample):
      return self.tree.traverse(sample).count
    
  def fit(self, X, y) -> None:
    # Preprocess the data
    y = self.preprocessing(X, y)
    
    # Compute the initial guess (which is like an average for binary classes)
    count = [0, 0]
    for binClass in y:
      count[binClass] += 1
    self.initialOdds = math.log(count[1] / count[0])
    self.initialGuess = 1 / (1 + math.exp(-self.initialOdds))
    self.probabilites = [[self.initialOdds, self.initialGuess]] * self.size
    
    # Fit the trees
    self.trees = []
    for treeIndex in range(self.maxNumTrees):
      tree = self.SmallDecisionTree(self)
      tree.fit(X, y)
      self.trees.append(tree)
    
  def predict(self, sample):
    total = 0
    for tree in self.trees:
      total += tree.predict(sample)
    # Compute the final odds and compare it to 0
    return int(self.initialGuess + self.learningRate * total > 0)
  
  def score(self, X, y) -> float:
    # Check the test data
    self.checkTestData(X, y)

    # And compute the score
    correctPredicted = 0
    for index, sample in enumerate(X):
      expected = int(y[index] == self.shrinked[1])
      correctPredicted += int(self.predict(sample) == expected)
    return float(correctPredicted / len(X))
