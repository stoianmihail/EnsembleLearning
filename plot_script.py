import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib as mpl
import peakutils
import numpy as np
import math

colourWheel =['b', 'r']

dashesStyles = [[3,1],[1000,1],[2,1,10,1],[4, 1, 1, 1, 1, 1]]
  
def main():
  with open("big_debug_rf.txt", "r") as input:
    confs = []
    curr = []
    trainErrs = []
    testErrs = []
    count = 0
    lastSize = 0
    for row in input:
      args = row.split(":")
      size = int(args[0])
      if lastSize != size:
        if trainErrs:
          confs.append((trainErrs, testErrs))
          trainErrs = []
          testErrs = []
        lastSize = size
      t = int(args[1])
      scores = args[2].strip().split(",")
      trainErrs.append(1 - float(scores[0]))
      testErrs.append(1 - float(scores[1]))
    if trainErrs:
      confs.append((trainErrs, testErrs))
      trainErrs = []
      testErrs = []
    plt.rc('font', family='serif')
    
    # Initialize the figure
    #plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    #plt.rc('xtick', labelsize=4)
    #plt.rc('ytick', labelsize=4)
    #plt.rc('axes', labelsize=4)
    fig, ax = plt.subplots(2, 2)
    
    linethick = 0.8
    alphaVal = 0.8
    numTrees = len(confs[0][0])
    xs = range(1, numTrees + 1)
    print(xs)
    print(len(confs))
    mylist = [1, 2, 3, 8]
    for index in range(4):
      curr = ax[int(index / 2), int(index % 2)]
      trainList = confs[mylist[index] - 1][0]
      testList = confs[mylist[index] - 1][1]
      
      # Find the right spot on the plot
      #plt.subplot(2,2, index + 1)

      for ptr in range(4):
        curr.plot(xs, confs[ptr][1], marker='', color='grey', linewidth=0.7, alpha=0.5)

      # Plot the lineplot
      curr.plot(xs, testList, marker='', color=palette(index + 1), linewidth=1.5, alpha=0.9, label="ρ = " + str(mylist[index]))
      #curr.plot(xs, trainList, marker='', color=palette(index + 8 + 1), linewidth=1.5, alpha=0.9, label=str(index))
    
      curr.legend(frameon=True, loc='upper right',ncol=1,handlelength=1)
    
      if int(index % 2) == 0 and int(index / 2) == 1:
        curr.set_ylabel('Test Error')
        curr.set_xlabel('Number of decision trees')
        curr.legend(frameon=True, loc='upper right',ncol=1,handlelength=1)
    
      #curr.set_xlim(0,1)
      #curr.set_ylim(0,1)


      if False:
        ax.plot(xs, trainList, color=palette(index), linestyle = 'dashed', dashes=dashesStyles[1], linewidth=1, marker='o', markersize=0.05, label="Training error", alpha=alphaVal)
      if False:
        ax.plot(xs, testList,
                color=palette(index),
                linestyle = 'dashed',
                dashes=dashesStyles[1],
                linewidth=1,
                marker='o',
                markersize=0.05,
                label="Test error",
                alpha=alphaVal)

    # Axis title
    #plt.text(0.5, 0.02, 'Time', ha='center', va='center')
    #plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')

    #indices = peakutils.indexes(-np.array(testScores), thres=0.1, min_dist=0)
    #ax.plot(np.array(xs)[indices[:7]],np.array(testScores)[indices[:7]], marker="o", color="purple", dashes=dashesStyles[0])

    #ax.set_xlabel('')
    #ax.set_ylabel('Error')
    #ax.set_xlabel('Number of decision trees')
    #ax.legend(frameon=True, loc='upper right',ncol=1,handlelength=1)
    #ax.grid(True, linestyle='dashed')
    #width=3.5
    #height=width/1.5
    #fig.set_size_inches(width, height)
    plt.savefig('plot_rf.png',dpi=500)
    plt.show()  
if __name__ == '__main__':
  main()
