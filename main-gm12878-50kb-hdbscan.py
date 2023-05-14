"""
思路：
1、将Hi-C接触矩阵看作是图的邻接矩阵
2、设计图自编码器（Graph Autoencoder）进行特征嵌入，得到嵌入表示
3、对嵌入表示进行聚类。
4、对聚类结果进行处理，得到TAD边界结果
5、对TAD边界进行可视化
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
import pandas as pd
import glob
"""
将矩阵划分为400x400子矩阵
"""
# 定义函数，用于将一个大矩阵分割成多个指定大小的子矩阵
def split_matrix(matrix, submatrix_size):
    # 获取大矩阵的大小
    n = matrix.shape[0]
    # 计算子矩阵的个数
    num_submatrices = int(np.ceil(n/submatrix_size))
    # 定义一个空列表，用于存储所有的子矩阵
    submatrices = []
    # 循环遍历每个子矩阵
    for i in range(num_submatrices):
        # 计算当前子矩阵在大矩阵中的起始索引和结束索引
        start_idx = i * submatrix_size
        end_idx = min((i+1) * submatrix_size, n)
        # 从大矩阵中提取当前子矩阵，并将其加入到子矩阵列表中
        submatrix = matrix[start_idx:end_idx, start_idx:end_idx]
        submatrices.append(submatrix)
    # 返回所有子矩阵组成的列表
    return submatrices

def diagonal_shift(matrix, shift):
    n = matrix.shape[0]
    shifted_matrix = np.zeros_like(matrix)
    for i in range(n):
        j = (i + shift) % n
        shifted_matrix[i, j] = matrix[i, j]
    return shifted_matrix

# 构建GAN模型
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
cellline="GM12878"
res=50000
metricList=["euclidean","manhattan","chebyshev"]
#Hi-C file dir
for res in [50000]:
    for chr in [7,10,18]:
        hic='/data2/ghy_data/ghy_data/GM12878/'+str(int(int(res)/1000))+'kb/chr'+str(chr)+'_kr_'+str(int(int(res)/1000))+'kb.npy'
        #sub_matrix dir
        matDir='/data1/ghy_data/toast/GM12878/lr=0.001-hdbscan/'+str(int(int(res)/1000))+'kb/chr'+str(chr)
        #result dir
        dir='/data1/ghy_data/toast/GM12878/lr=0.001-hdbscan/'+str(int(int(res)/1000))+'kb/chr'+str(chr)
        # 创建多级目录
        if not os.path.exists(matDir):
            os.makedirs(matDir)
        else:
            print(f"Directory '{matDir}' already exists.")
        #quality file 
        qualityFile=dir+'/Toast_quality-metrics-'+str(int(int(res)/1000))+'kb_chr'+str(chr)+'.txt'
        matrix = np.load(hic)
        num_rows, num_cols = matrix.shape
        print(num_cols)
        print(num_rows)
        num = (num_cols - 1) // 400 + 1
        print("number of submatrix:",num)
        # 划分子矩阵
        sub_matrices = split_matrix(matrix,400)
        # 保存为npy格式
        np.save(matDir+'/sub_matrices.npy', sub_matrices)
        # 加载保存的子矩阵
        #sub_matrices = np.load(matDir+'/sub_matrices.npy')

        #保存所有子矩阵
        for i in range(len(sub_matrices)):
            matFile=matDir+'/sub_matrix_{}.txt'.format(i)
            # 保存为txt格式
            np.savetxt(matFile, sub_matrices[i], fmt='%f')

            # 读取400x400的对称正定矩阵
            # 加载邻接矩阵
            adj_matrix = np.loadtxt(matFile)
            submatricNumber=i
            # 随机初始化节点特征向量
            feature_matrix = torch.eye(400)
            # 训练图自编码器并得到节点特征嵌入
            for lr in [0.001]:
                embedding = train_gae(adj_matrix,lr)
                # 对节点特征进行HDBSCAN聚类
                clusterer = hdbscan.HDBSCAN(metric="euclidean", gen_min_span_tree=True)
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
        # 待处理文件所在目录
        directory = matDir+'/'
        newdir=directory+"/merge/"
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        else:
            print(f"Directory '{newdir}' already exists.")
        # 循环读取文件名
        for filename in os.listdir(directory):
            # 筛选出以".tad"结尾的文件
            if filename.endswith(".tad"):
                # 从文件名中提取序号
                index = int(filename.split("_")[2])
                # 读取文件内容
                df = pd.read_csv(directory+filename, header=None, delim_whitespace=True)
                # 修改数据
                df.iloc[:, 0] += 400 * index
                df.iloc[:, 2] += 400 * index
                df.iloc[:, 1] += 400 * index * res
                df.iloc[:, 3] += 400 * index * res
                # 构造新文件名
                new_filename = str(index)+".tad"
                # 写入新文件
                df.to_csv(newdir+new_filename, sep="\t", header=False, index=False)

        # 拼接文件
        files = os.listdir(newdir)
        print(files)
        basename = os.path.basename(files[1])
        print(basename)
        name_without_ext = os.path.splitext(basename)[0]  # 去掉扩展名
        print(name_without_ext)
        # 按照序号排序文件名
        files = sorted(files, key=lambda x: int(x.split('.')[0]))

        # 合并文件内容
        merged_content = ''
        for file_name in files:
            file_path = os.path.join(newdir, file_name)
            with open(file_path, 'r') as f:
                content = f.read().strip()
                merged_content += content + '\n'

        # 将合并后的内容写入merged_TAD.tad文件
        with open(newdir+'chr'+str(chr)+'merged_TAD.tad', 'w') as f:
            f.write(merged_content)


