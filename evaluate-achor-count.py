import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import heapq
import argparse
import math
import matplotlib as mpl
import hdbscan
import datetime
import sklearn.metrics as metrics
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
def boundaryPlot(labels):
    n = len(labels)
    boundary = np.zeros(n)
    i = 0
    label = -1
    start = 0
    while i < n:
        if labels[i] == label:
            boundary[i] = start
        else:
            start = i
            label = labels[i]
            boundary[i] = i
        i = i + 1
    return boundary

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

def evalTAD(hicfile,TAD1,TAD2):
    start1, end1 = readTAD(TAD1)
    start2, end2 = readTAD(TAD2)
    print(len(start1))
    print(len(start2))
    label1=getLabel(hicfile,start1,end1)
    label2=getLabel(hicfile,start2,end2)
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
    filename = "/data1/ghy_data/toast/GM12878/chip-seq/{}.bed".format(factorname)
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
def draw_heat_map(df, mask_data, rx_tick, sz_tick, sz_tick_num, rx_tick_num, x_label, z_label, map_title):
    # 用于画图
    c_map = sns.cubehelix_palette(start=1.6, light=0.8, as_cmap=True, reverse=True)
    plt.subplots(figsize=(6, 6))
    ax = sns.heatmap(df, vmax=600, vmin=500, mask=mask_data, cmap=c_map,
                     square=True, linewidths=0.005, xticklabels=rx_tick, yticklabels=sz_tick)

    ax.set_xticks(rx_tick_num)
    ax.set_yticks(sz_tick_num)

    ax.set_xlabel(x_label)
    ax.set_ylabel(z_label)
    ax.set_title(map_title)
    plt.savefig(map_title + '.png', dpi=300)
    plt.show()
    plt.close()
# 设置颜色
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)


res = 50000
cellline="GM12878"
dir="/home/ghaiyan/project/toast/TADResults/GM12878/50kb/IS"
chrs=[1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,19,20,21,22]
#chrs=[7,10,18]
factors=["ctcf","smc3","rad21","polr2a","H3K9me3","h3k4me3","H3K4me1","H3K36me3","promoter-GM12878","enhancer-GM12878"]
for chr in chrs:
    hicfile='/data2/ghy_data/ghy_data/GM12878/'+str(int(int(res)/1000))+'kb/chr'+str(chr)+'_kr_'+str(int(int(res)/1000))+'kb.npy'
    hic=np.load(hicfile)
    tadfile=dir+"/chr"+str(chr)+"_"+str(res)+"IS.tad"
    TADquality=tadQuality(tadfile,hic)
    for factor in factors:
        start, end = readTAD(tadfile)
        print("length of TADs",len(start))
        lentad=len(start)
        ctcf=getctcf(factor,chr)
        tadlistctcf = getlist(tadfile,ctcf)
        achorctcf=np.array(tadlistctcf)
        achorCount,countratio=getCount(achorctcf)
        qualityFile=dir+"/quality-metrics-gm12878-50kb-lr0.001.txt"
        with open(qualityFile,'a+') as f:
            f.write("\t".join(("chr",str(chr),str(factor),str(lentad),str(TADquality),str(achorCount),str(countratio)))+'\n')
            f.close()
    