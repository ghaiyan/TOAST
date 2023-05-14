import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
#plot TAD boundaries
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

def plot_TAD(hic,tadfile,dir,dataname):
    start, end = readTAD(tadfile)
    lentad=len(start)
    palette=sns.color_palette("bright",10)
    #print(labels)
    plt.figure(figsize=(10.5,10))
    start1=3600
    end1=3800
    sns.heatmap(data=hic[start1:end1, start1:end1], robust=True,cmap="OrRd")
    for i in range(0,lentad):
        if start1<start[i]<end1 and start1<end[i]<end1:
            #print(start[i])
            plt.hlines(y=start[i]-start1,xmin=start[i]-start1,xmax=end[i]-start1)
            plt.vlines(x=end[i]-start1,ymin=start[i]-start1,ymax=end[i]-start1)
    plt.title('TAD boundary')
    #plt.savefig(dir+"/"+methodname+"1_399.png", dpi=300)
    plt.savefig(dir+"/"+dataname+"_"+"3600_3800.pdf", format='pdf', bbox_inches='tight')
    plt.show()
#mian code
"""
cellline="simulation"
res=40000
chr="sim"
#Hi-C file dir
dir="/home/ghaiyan/project/toast/TADResults/differentNoiseResults"
todir="/home/ghaiyan/project/toast/TADResults/differentNoiseResults/heatmaps"
methods=["","__IC_Finder","Caspian","clusterTAD","HiCseg","IS","TopDom"]
for noise in [4,8,12,16,20]:
    hic='/home/ghaiyan/project/CASPIAN/SimulationData/'+str(noise)+'noise.hic'
    hicmatrix=np.loadtxt(hic)
    
    for method in methods:
        tadfile=dir+"/{}noise{}.tad".format(noise,method)
        dataname=str(noise)+'noise_'+str(method)
        plot_TAD(hicmatrix,tadfile,todir,dataname)

"""
cellline="GM12878"
res=50000
chr=1
#Hi-C file dir
dir="/home/ghaiyan/project/toast/TADResults/GM12878/50kb/"
todir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb/heatmaps"
methods=["toast","IC-Finder","Caspian","clusterTAD","HiCseg","IS","Topdom","DI"]
hicfile='/data2/ghy_data/ghy_data/GM12878/50kb/chr1_kr_50kb.npy'
hicmatrix=np.load(hicfile)   
for method in methods:
    tadfile=dir+method+"/chr"+str(chr)+"_50000"+str(method)+".tad"
    dataname="chr"+str(chr)+"_50000"+str(method)
    plot_TAD(hicmatrix,tadfile,todir,dataname)