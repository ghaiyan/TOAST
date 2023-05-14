"""
1.Treat the Hi-C contact matrix as the adjacency matrix of a graph.
2.Design a Graph Autoencoder (GAE) to perform feature embedding and obtain the embedding representation.
3.Cluster the embedding representation.
4.Process the clustering results to obtain TAD boundary results.
5.Visualize the TAD boundaries.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import matplotlib as mpl
import seaborn as sns
import heapq
import argparse
import math
from sklearn import metrics
import hdbscan
import os
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
"""
将矩阵划分为400x400子矩阵
"""

def split_matrix(matrix, submatrix_size):
    
    n = matrix.shape[0]
   
    num_submatrices = int(np.ceil(n/submatrix_size))
    
    submatrices = []
    
    for i in range(num_submatrices):
        
        start_idx = i * submatrix_size
        end_idx = min((i+1) * submatrix_size, n)
        
        submatrix = matrix[start_idx:end_idx, start_idx:end_idx]
        submatrices.append(submatrix)
    
    return submatrices


class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, adj_matrix):
        embedding = self.encoder(adj_matrix)
        adj_hat = self.decoder(embedding)
        return embedding, adj_hat

def train_gae(adj_matrix,lr):
    input_dim = adj_matrix.shape[0]
    hidden_dim = 64
    embedding_dim = 16
    num_epochs = 200

    model = GAE(input_dim, hidden_dim, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    adj_matrix = torch.FloatTensor(adj_matrix)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        _, adj_hat = model(adj_matrix)
        loss = criterion(adj_hat, adj_matrix)
        loss.backward()
        optimizer.step()

    embedding, _ = model(adj_matrix)
    return embedding.detach().numpy()



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

#get TAD file
def getTAD(tadfile,cellline,chr,submatricNumber,res,label):
    boundaries=boundaryPlot(label)
    print(boundaries)
    
    i = 0
    with open(tadfile, "w") as out:
        while i < len(boundaries):
            if boundaries[i] < i:
                start = i - 1
                while i<len(boundaries) and boundaries[i] == start:
                    end = i
                    i = i + 1
                if end-start>=(2):
                    startbin = start * res
                    endbin = end * res
                    out.write("\t".join((str(start), str(startbin), str(end), str(endbin))) + "\n")
                else:
                    start=start-1
            i = i + 1
        out.close()

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


def evalTAD(hicfile,TAD1,TAD2):
    start1, end1 = readTAD(TAD1)
    start2, end2 = readTAD(TAD2)
    print(len(start1))
    print(len(start2))
    label1=getLabel(hicfile,start1,end1)
    label2=getLabel(hicfile,start2,end2)
    RI=metrics.rand_score(label1, label2)   
    FMS=metrics.fowlkes_mallows_score(label1, label2)
    return RI, FMS

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
    filename = "/home/ghaiyan/project/CASPIAN/chip-seq/GM12878/{}.bed".format(factorname)
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

import pandas as pd
#mian code
#split matrix
cellline="simulation"
res=40000
chr="sim"
#Hi-C file dir
for noise in [150]:
    for alpha in [200]:
        hic='/data1/ghy_data/toast/simulation/simulated_TAD/'+str(noise)+'_TADlike_alpha_'+str(alpha)+'_set6.mat'
        #sub_matrix dir
        matDir='/data1/ghy_data/toast/simulation/simulated_TAD/'+str(noise)+"_"+str(alpha)
        #result dir
        dir='/data1/ghy_data/toast/simulation/simulated_TAD/'+str(noise)+"_"+str(alpha)
        if not os.path.exists(matDir):
            os.makedirs(matDir)
        else:
            print(f"Directory '{matDir}' already exists.")
        #quality file 
        qualityFile=dir+'/Toast_quality-metrics-'+str(noise)+str(alpha)+'.txt'
        matrix = np.loadtxt(hic)
        num_rows, num_cols = matrix.shape
        print(num_cols)
        print(num_rows)
        num = (num_cols - 1) // 400 + 1
        print(num)
        sub_matrices = split_matrix(matrix,400)
        np.save(matDir+'/sub_matrices.npy', sub_matrices)
        for i in range(len(sub_matrices)):
            matFile=matDir+'/sub_matrix_{}.txt'.format(i)        
            np.savetxt(matFile, sub_matrices[i], fmt='%f')                        
            adj_matrix = np.loadtxt(matFile)
            submatricNumber=i
            feature_matrix = torch.eye(400)
            for lr in [0.001]:
                embedding = train_gae(adj_matrix,lr)
                clusterer = hdbscan.HDBSCAN(metric="euclidean",gen_min_span_tree=True)
                node_labels = clusterer.fit_predict(embedding)
                print(node_labels)
                #get TAD file
                tadfile=dir+"/{}_chr{}_{}_{}_{}.tad".format(cellline,chr,submatricNumber, res,lr)
                getTAD(tadfile,cellline,chr,submatricNumber,res,node_labels)
                #plot TADs
                plot_TAD(adj_matrix,tadfile)
                quality=tadQuality(tadfile,adj_matrix)
                print(quality)
                start, end = readTAD(tadfile)
                lentad=len(start)
                with open(qualityFile,'a+') as f:
                        f.write("\t".join((str(submatricNumber),str(lr),str(lentad),str(quality)))+'\n')
                        f.close()
        
        directory = matDir+'/'
        newdir=directory+"/merge/"
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        else:
            print(f"Directory '{newdir}' already exists.")       
        for filename in os.listdir(directory):           
            if filename.endswith(".tad"):                
                index = int(filename.split("_")[2])                
                df = pd.read_csv(directory+filename, header=None, delim_whitespace=True)
                
                df.iloc[:, 0] += 400 * index
                df.iloc[:, 2] += 400 * index
                df.iloc[:, 1] += 400 * index * res
                df.iloc[:, 3] += 400 * index * res
                
                new_filename = str(index)+".tad"
                
                df.to_csv(newdir+new_filename, sep="\t", header=False, index=False)

        # merge files
        files = os.listdir(newdir)
        print(files)
        basename = os.path.basename(files[1])
        print(basename)
        name_without_ext = os.path.splitext(basename)[0]  # 去掉扩展名
        print(name_without_ext)
        
        files = sorted(files, key=lambda x: int(x.split('.')[0]))

        merged_content = ''
        for file_name in files:
            file_path = os.path.join(newdir, file_name)
            with open(file_path, 'r') as f:
                content = f.read().strip()
                merged_content += content + '\n'
        with open(newdir+'chr'+str(chr)+'_'+str(res)+'toast.tad', 'w') as f:
            f.write(merged_content)

