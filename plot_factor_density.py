import bbi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.neighbors import KernelDensity as kde
import numpy as np

mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
dir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb/factor-density-plot"
factordir="/data1/ghy_data/toast/GM12878/chip-seq/"
for tadname in ["CTCF","H3K4me3","H3K27ac","POLR2A","H3K27me3","H3K9me3"]:
    boundaries = pd.read_excel("/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb/toast-50kb-tad.xlsx")
    factorfile=factordir+tadname+'.bigWig'
    print(boundaries.head())
    flank = 40000 # Length of flank to one side from the boundary, in basepairs
    nbins = 100 # Number of bins to split the region
    resolution=50000
    stackup = bbi.stackup(
        factorfile,
        boundaries.chrom,
        boundaries.start+resolution//2-flank,
        boundaries.end+resolution//2+flank,
        bins=nbins)
    print(stackup)
    f, ax = plt.subplots(figsize=[7,5])
    
    ax.plot(np.nanmean(1-stackup, axis=0))
    
    ax.set(xticks=np.arange(0, nbins+1, 10),
        xticklabels=(np.arange(0, nbins+1, 10)-nbins//2)*flank*2/nbins/1000,
        xlabel='Distance from boundary, kbp',
        ylabel=tadname+' ChIP-Seq mean P value',
        title=tadname);
    plt.savefig(dir+'/GM12878_tad-toast_{}-Pvalue.pdf'.format(tadname), format='pdf', bbox_inches='tight')
    plt.savefig(dir+'/GM12878_tad-toast_{}-Pvalue.png'.format(tadname), dpi=300)
    plt.show()
