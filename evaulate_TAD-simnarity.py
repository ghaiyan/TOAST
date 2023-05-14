import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import matplotlib as mpl
import seaborn as sns
import heapq
import argparse
import math
from sklearn import metrics
import os
import pandas as pd
import glob

mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42


def readTAD(tadfile):
    #tads = "/home/ghaiyan/project/CASPIAN/evaluate_TADS/GM12878/chr19_5kb/TAD/{}.txt".format(tadsname)
    f = open(tadfile)
    line=f.readline()
    start=[]
    end=[]
    while line:
        line = line.split()
        start1 = int(line[0])
        end1 = int(line[2])
        start.append(start1)
        end.append(end1)
        line=f.readline()
    f.close()
    return start, end

def getLabel(hicfile,start, end):
    hic=np.load(hicfile)
    n = len(hic)
    labels = np.zeros(n)
    for i in range(n):
        labels[i] = 0
    for j in range(len(start)+1):
        s=start[j-1]
        m=end[j-1]
        labels[s]=2
        labels[m]=2
        for k in range(s+1,m):
            labels[k]=1
    return labels

#评估俩TAD之间的相似性
def evalTAD(hic,TAD1,TAD2):
    start1, end1 = readTAD(TAD1)
    start2, end2 = readTAD(TAD2)
    print(len(start1))
    print(len(start2))
    label1=getLabel(hic,start1,end1)
    label2=getLabel(hic,start2,end2)
    AMI=metrics.adjusted_mutual_info_score(label1, label2)
    RI=metrics.rand_score(label1, label2)
    AR=metrics.adjusted_rand_score(label1, label2)
    HS=metrics.homogeneity_score(label1, label2)
    VMS=metrics.v_measure_score(label1, label2)
    FMS=metrics.fowlkes_mallows_score(label1, label2)
    return AMI, RI, AR, HS, VMS, FMS

def calcuDist(arr, e):
    size = len(arr)
    idx = 0
    val = abs(e - arr[idx][0])
    
    for i in range(1, size):
        val1 = abs(e - arr[i][0])
        if val1 < val:
            idx = i
            val = val1
    if arr[idx][0] < e and arr[idx][1] < e:
        return e - arr[idx][1]
    elif arr[idx][0] < e and arr[idx][1] > e:
        return 0
    else:
        return e - arr[idx][0]


def getlist(tadfile,ctcf):
    #tad = "/home/ghaiyan/project/CASPIAN/evaluate_TADS/GM12878/chr19_5kb/TAD/{}.txt".format(name)  
    #tad = "/home/ghaiyan/project/CASPIAN/evaluate_TADS/GM12878/chr19_5kb/TAD/compare/{}.txt".format(name)
    distances = []
    with open(tadfile) as tad:
        for num, line in enumerate(tad):
            line = line.split()
            start = int(line[1])
            end = int(line[2])
            dist_start = calcuDist(ctcf, start)
            dist_end = calcuDist(ctcf, end)
            if abs(dist_start) <= abs(dist_end):
                distances.append(dist_start)
            else:
                distances.append(dist_end)
        tad.close()
    return list(set(distances))

def getctcf(factorname,chr):
    #ctcf。。。
    filename = "/data2/ghy_data/ghy_data/GM12878/chip-seq/{}.bed".format(factorname)
    ctcf=[]
    with open(filename, 'r') as file_to_read:
        for i, line in enumerate(file_to_read):
            line = line.strip().split()
            chrname="chr"+str(chr)
            if line[0] == chrname:
                ctcf.append([int(line[1]), int(line[2])])
        file_to_read.close()
    return ctcf

def getCount(tadlist):
    count=0
    i=0
    for i in range(len(tadlist)):

        if(abs(tadlist[i])<20000):
            count=count+1
            i=i+1
        else:
            i=i+1
    print(count)
    countratio=count/len(tadlist)
    return count,countratio

#plot TAD boundaries
def plot_TAD(hic,tadfile):
    start, end = readTAD(tadfile)
    lentad=len(start)
    palette=sns.color_palette("bright",10)
    #print(labels)
    plt.figure(figsize=(10.5,10))
    start1=1
    end1=399
    sns.heatmap(data=hic[start1:end1, start1:end1], robust=True,cmap="OrRd")
    for i in range(0,lentad):
        if start1<start[i]<end1 and start1<end[i]<end1:
            #print(start[i])
            plt.hlines(y=start[i]-start1,xmin=start[i]-start1,xmax=end[i]-start1)
            plt.vlines(x=end[i]-start1,ymin=start[i]-start1,ymax=end[i]-start1)
    plt.title('TAD boundary')
    plt.savefig(tadfile+".png", dpi=300)
    plt.savefig(tadfile+".pdf", format='pdf', bbox_inches='tight')
    plt.show()

def tadQuality(tadFile,hic):
    """TAD quality"""
    n = len(hic)
    tad = np.loadtxt(tadFile)
    intra = 0
    intra_num = 0
    for n in range(len(tad)):
        for i in range(int(tad[n,0]),int(tad[n,2]+1)):
            for j in range(int(tad[n,0]),int(tad[n,2]+1)):
                intra = intra + hic[i,j]
                intra_num = intra_num + 1

    if intra_num!=0:
        intra = intra / intra_num
        print("intra TAD: %0.3f" % intra)
    else:
        intra = 0
    
    inter = 0
    inter_num = 0
    for n in range(len(tad) - 1):
        for i in range(int(tad[n,0]),int(tad[n,2]+1)):
            for j in range(int(tad[n+1,0]),int(tad[n+1,2]+1)):
                inter = inter + hic[i,j]
                inter_num = inter_num + 1
    if inter_num!=0:
        inter = inter / inter_num
        print("inter TAD: %0.3f" % inter)
    else:
        inter = 0
    print("quality: %0.3f" % (intra - inter))
    quality=intra - inter
    return quality


#mian code
#split matrix
#modify the following parameters:
"""
cellline
res
number of chr
folder of Hi-C matrix
folder for stroing results
"""
dir="/home/ghaiyan/project/toast/TADResults/differentNoiseResults/lr=0.001-hdbscan/"
res = 50000
cellline="Gm12878"
#methods=["toast-raw","IC-Finder","Caspian","clusterTAD","HiCseg","IS","Topdom"]
methods=["DI"]
chrs=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
for chr in chrs:
    hicfile='/data2/ghy_data/ghy_data/GM12878/'+str(int(int(res)/1000))+'kb/chr'+str(chr)+'_kr_'+str(int(int(res)/1000))+'kb.npy'
    hic=np.load(hicfile)
    print("start to read Hi-C matrix")
    dir='/home/ghaiyan/project/toast/TADResults/GM12878/50kb/'
    trueTAD="/home/ghaiyan/project/toast/TADResults/GM12878/50kb/toast-kr/chr"+str(chr)+"_"+str(res)+"toast.tad"
    for method in methods: 
        tadfile=dir+str(method)+"/chr"+str(chr)+"_"+str(res)+str(method)+".tad"      
        AMI, RI, AR, HS, VMS, FMS=evalTAD(hicfile,tadfile,trueTAD)
        qualityFile1=dir+"quality-metrics-Toast-simlarity.txt"
        with open(qualityFile1,'a+') as f:
            f.write("\t".join((str(chr),str(method),str(RI),str(FMS)))+'\n')
            f.close()
