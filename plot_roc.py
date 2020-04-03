import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.integrate import quad


from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


colourWheel =['purple', 'teal']

dashesStyles = [[3,1],[1000,1],[2,1,10,1],[4, 1, 1, 1, 1, 1]]

with open("debug_roc.txt", "r") as input:
  ys1 = []
  ys2 = []
  sizes = []
  for row in input:
    args = row.split(":")
    randomSize = int(args[0])
    sizes.append(randomSize)
    coords = args[1].split(",")
    specifity = 1 - float(coords[0])
    sensitivity = float(coords[1])
    
    ys1.append(specifity)
    ys2.append(sensitivity)

  #xs = np.array([0] + xs + [1])
  #ys = np.array([0] + ys + [1])
  
  fig, ax = plt.subplots()

  linethick = 1.0
  alphaVal = 0.8
  for index, y in enumerate(ys1):
    ax.plot(sizes[index], y, linestyle="None", marker="o", markersize=int(y**6 * 50), color="black")
  for index, y in enumerate(ys2):
    ax.plot(sizes[index], y, linestyle="None", marker="o", markersize=int(y**10 * 80), color="red")
  
  ax.set_xlabel('Number of randomly selected features (ρ)')
  ax.set_ylabel('Magnitude')
  ax.plot(sizes, ys2, linestyle="dashed", label="Sensitivity", dashes=dashesStyles[0], color=colourWheel[1])
  ax.plot(sizes, ys1, linestyle="dotted", label="Specificity", color=colourWheel[0])
  ax.set_ylim([max(min(np.array(ys1).min(), np.array(ys2).min()) - 0.05, 0), min(max(np.array(ys1).max(), np.array(ys2).max()) + 0.05, 1)])
  ax.legend(frameon=True, loc='upper right',ncol=2,handlelength=1)
  plt.savefig('proximity.png',dpi=500)
  plt.show()
  if False:
      
    plt.plot(sizes, ys1,
                color=colourWheel[0],
                linestyle = 'dotted',
                #dashes=dashesStyles[1],
                #linewidth=linethick,
                marker='o',
                markersize=0.05,
                label="Specifity",
                alpha=alphaVal)
    
    plt.plot(sizes, ys2,
                color=colourWheel[1],
                linestyle = 'dotted',
                #dashes=dashesStyles[1],
                #linewidth=linethick,
                marker='o',
                markersize=0.05,
                label="Sensitivity",
                alpha=alphaVal)
    #plt.plot(sizes, y, linestyle="dotted", color="red")

    
    
    print(xs)
    print(ys)
    # construct a linear interpolation function of the data set 
    f_interp = lambda xx: np.interp(xx, xs, ys)
    
    area, err = quad(f_interp, xs.min(), xs.max(), points=xs)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lw = 2

    plt.plot(xs, ys, color='darkorange',
          lw=lw, label='ROC curve (area = %0.2f)' % area)
    plt.plot([xs.min(), xs.max()], [ys.min(), ys.max()], color='navy', lw=lw, linestyle='--')
    
    index = -1
    for xy in zip(xs, ys):
      if index != -1 and index != len(sizes):
        ax.annotate(str(sizes[index]), xy=xy, textcoords='data')
      index += 1
    
    plt.xlim([xs.min(), xs.max()])
    plt.ylim([ys.min(), ys.max()])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
