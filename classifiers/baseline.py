# Baseline for each classifier we are going to implement
class Classifier:
  def __init__(self,):
    pass
  
  def __repr__(self):
    pass
  
  def preprocessing(self, X, y, storeType = "withoutLists"):
    # Shrink y and sort the values, since we assume that the lower value represents the minus class
    shrinked = sorted(list(set(y)))
    if len(shrinked) == 1:
      print("Your training data is redundant. All samples fall into the same class")
      sys.exit(1)
    if len(shrinked) > 2:
      print("We can support by now only binary classification")
      sys.exit(1)
    
    self.size = len(X)
    self.numFeatures = len(X[0])
    
    # Translate y into binary classes (-1, 1)
    z = []
    for indexInData in range(self.size):
      z.append(int(y[indexInData] == shrinked[1]))
    y = z
  
    # Build up the container
    self.store = []
    for feature in range(self.numFeatures):
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
    for feature in range(self.numFeatures):
      size = len(self.store[feature])
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
    
    # Get rid of values, which do not affect the accuracy (this mainly applies for continous variables)
    for feature in range(self.numFeatures):
      if self.types[feature] == "continous":
        size = len(self.store[feature])
        tmp = sorted(self.store[feature].items())
        save = []
        acc = [[], []]
        for ptr, (value, [minusList, plusList]) in enumerate(tmp):
          acc[0] += minusList
          acc[1] += plusList
          if ptr == size - 1:
            if storeType == "withoutLists":
              save.append(value + 1)
            else:
              save.append((value + 1, [acc[0], acc[1]]))
          else:
            nextValue = tmp[ptr + 1][0]
            nextMinusList = tmp[ptr + 1][1][0]
            nextPlusList = tmp[ptr + 1][1][1]
            
            if not ((not minusList and not nextMinusList) or (not plusList and not nextPlusList)):
              mid = (value + nextValue) / 2
              if storeType == "withoutLists":
                save.append(mid)
              else:
                save.append((mid, [acc[0], acc[1]]))
                acc[0] = []
                acc[1] = []
        self.store[feature] = save
      else:
        if storeType == "withoutLists":
          self.store[feature] = sorted(self.store[feature].keys())
        else:
          self.store[feature] = sorted(self.store[feature].items())
    return y
  
  def fit(self, X, y) -> None:
    pass
  
  def predict(self, sample):
    pass
  
  def checkTestData(self, X, y):
    if len(X) != len(y):
      print("x-axis and y-axis of test data do not have the same size")
      sys.exit(1)
    self.size = len(X)
    if not self.size:
      print("Empty test data!")
      sys.exit(1)
    
    # Define the binary class
    self.shrinked = sorted(list(set(y)))
    if len(self.shrinked) == 1:
      print("Your test data is redundant. All samples fall into the same class")
      sys.exit(1)
    elif len(self.shrinked) > 2:
      print("We can support by now only binary classification")
      sys.exit(1)
    pass
  
  def score(self, X, y) -> float:
    pass
