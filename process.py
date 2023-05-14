import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse

def contactToDist(contactMat, alpha):
    """Convert contact matrix to distance matrix."""
    distMat = np.zeros_like(contactMat)
    numRows = len(contactMat)
    for i in range(numRows):
        for j in range(numRows):
            if contactMat[i,j] != 0:
                distMat[i,j] = contactMat[i,j]**(alpha)
    return distMat

def main():
    hic=np.loadtxt("chr19_50000.hic")
    dist = contactToDist(hic, 0.2)
    pic = pd.DataFrame(dist)
    plt.figure()
    sns.heatmap(data=pic)
    plt.show()


if __name__ == "__main__":
    main()
