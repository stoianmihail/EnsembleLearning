import sys
import random

def printData(data):
# Print line-by-line the data
  for (f, l) in data:
    print(str(f) + " -> " + str(l))
  pass

def readData(fileName):
# Read the data from the file 'fileName'
  with open(fileName, 'r') as file:
    data = []
    for line in file:
      args = line.split(',')
      size = len(args)
      curr = args[:(size - 1)]
      label = args[size - 1]
      data.append((curr, label))
    random.shuffle(data)
    return data
  
def main(fileName):
  data = readData(fileName)
  printData(data)
    
if __name__ == '__main__':
  main(sys.argv[1])
