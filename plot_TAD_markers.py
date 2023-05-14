import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pyBigWig
H3K9me3 = pyBigWig.open("/data1/ghy_data/toast/GM12878/chip-seq/H3K9me3.bigWig")
H3K27ac = pyBigWig.open("/data1/ghy_data/toast/GM12878/chip-seq/H3K27ac.bigWig")
H3K4me3 = pyBigWig.open("/data1/ghy_data/toast/GM12878/chip-seq/H3K4me3.bigWig")
CTCF=pyBigWig.open("/data1/ghy_data/toast/GM12878/chip-seq/CTCF.bigWig")
H3K27me3=pyBigWig.open("/data1/ghy_data/toast/GM12878/chip-seq/H3K27me3.bigWig")
POLR2A=pyBigWig.open("/data1/ghy_data/toast/GM12878/chip-seq/POLR2A.bigWig")
GENES = pd.read_table("/data1/ghy_data/toast/GM12878/chip-seq/hg38_gc_cov_100kb.tsv")
resolution=50000
def plot_bwTrack(ax, bw, ylabel, chrom, start, end, resolution=20000 , yminx=5, ymaxx=95,rotation=0, fl='%0.2f',color
='#464451'):
    ax.tick_params(bottom=False,top=False,left=True,right=False)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_color('none')
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_color('none')
    ax.spines['right'].set_linewidth(0)

    x = int((end-start)/resolution)

    plot_list = bw.stats(chrom, start, end, type="mean", nBins=x)
    plot_list = [0 if v is None else v  for v in plot_list]
    
    width = 1
    ax.bar(x=range(0,x), height=plot_list, width=1, bottom=[0]*(x),color=color,align="edge",edgecolor=color)    
    ax.set_xlim(0,x)

    #ax.set_xticks([])
    #ax.set_xticklabels([])

    ymin = np.percentile(plot_list,yminx)
    ymax = np.percentile(plot_list,ymaxx)

    ax.set_yticks([ymin, ymax])
    ax.set_yticklabels([fl % ymin, fl % ymax], fontsize=7)
    
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=8, rotation=rotation, horizontalalignment='right',verticalalignment='center')
    
    ax.set_xticks([])
    ax.set_xticklabels('')

def two_degree_bc(x_l=10, x_r=90, y_lr=0, y2=10, dots_num=100): #bezier curve
    xt = []
    yt = []
    x_mid = (x_l + x_r)/2
    x_dots12 = np.linspace(x_l, x_mid, dots_num)
    y_dots12 = np.linspace(y_lr, y2, dots_num)
    x_dots23 = np.linspace(x_mid, x_r, dots_num)
    y_dots23 = np.linspace(y2, y_lr, dots_num)
    for i in range(dots_num):
        x = x_dots12[i] + (x_dots23[i]-x_dots12[i])*i / (dots_num-1)
        y = y_dots12[i] + (y_dots23[i]-y_dots12[i])*i / (dots_num-1)
        xt.append(x)
        yt.append(y)
    return (xt, yt)

def plot_loop(ax, ylabel, file_path, chrom: int, start: int, end: int, FDR=0.05):
    if file_path.endswith('.npy'):
        loops = np.load(file_path, allow_pickle=True).item()
        loops = loops[chrom]
        loops = list(filter(lambda x: (x[0]>=start)&(x[1]<=end), loops))
        lengths = [loop[1]-loop[0] for loop in loops]
    else:
        loops = pd.read_table(file_path)
        loops = loops[loops['BIN1_CHR'].apply(str)==str(chrom)]
        loops = loops[loops['FDR'] <= FDR]
        lengths = loops["BIN2_START"] - loops["BIN1_START"]

        loops = loops[
            ((loops['BIN1_START'] >= start) & (loops['BIN1_START'] <= end)) |
            ((loops['BIN2_START'] >= start) & (loops['BIN2_START'] <= end) |
             (loops['BIN1_END'] >= start) & (loops['BIN1_END'] <= end)) |
            ((loops['BIN2_END'] >= start) & (loops['BIN2_END'] <= end))]
        loops = loops.apply(lambda x: (x['BIN1_START'], x['BIN2_START']), axis=1).values.tolist()

    if len(loops) == 0:
        for i in ['top', 'right', "left", "bottom"]:
            ax.spines[i].set_color('none')
            ax.spines[i].set_linewidth(0)
        return

    top_y = 0
    for loop, length in zip(loops, lengths):
        top = length / max(lengths)
        top = max(0.5, top)
        top = min(0.8, top)
        
        xt, yt = two_degree_bc(x_l=loop[0], x_r=loop[1], y_lr=0, y2=top, dots_num=100)
        
        ax.plot(xt, yt, color='#66AC84')
        if max(yt) > top_y:
            top_y = max(yt)

    ax.set_xlim(start,end)
    ax.set_ylim(0,0.5)
    ax.set_ylabel(ylabel, fontsize=8, rotation=0, horizontalalignment='right',verticalalignment='center')
    for i in ['top', 'right']:
        ax.spines[i].set_color('none')
        #ax.spines[i].set_linewidth(0.5)
        
    ax.spines["bottom"].set_color('black')
    ax.spines["bottom"].set_linewidth(1)  
    
    
    ax.tick_params(bottom =False,top=False,left=False,right=False)
    ax.set_xticklabels("")
    ax.set_yticklabels("")

def plot_gene(ax, gene_bed, ylabel, chrom, start, end, resolution, rotation=0, fl='%0.2f',color
='#464451'):
    ax.tick_params(bottom=False,top=False,left=True,right=False)
    ax.spines['left'].set_color('k')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('k')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['top'].set_color('none')
    ax.spines['top'].set_linewidth(0)
    ax.spines['right'].set_color('none')
    ax.spines['right'].set_linewidth(0)

    x = int((end-start)/resolution)
    
    plot_list = []
    for i in range(x):
        _start = int(int(start/resolution)*resolution) + resolution * i
        plot_list.append(
            gene_bed[
                (gene_bed['chrom']==chrom)&
                (gene_bed['start']>=_start)&
                (gene_bed['end']<=_start+resolution)]['geneNum'].sum())
    
    width = 1
    ax.bar(x=range(0,x), height=plot_list, width=1, bottom=[0]*(x),color=color,align="edge", edgecolor=color)    
    ax.set_xlim(0,x)

    ymin, ymax = min(plot_list), max(plot_list)
    
    ax.set_yticks([ymin, ymax])
    ax.set_yticklabels([fl % ymin, fl % ymax], fontsize=7)
    
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=8, rotation=rotation, horizontalalignment='right',verticalalignment='center')
    
    ax.set_xticks([])
    ax.set_xticklabels('')

def show(chrom, start, end, resolution, FDR=0.05):
    fig = plt.figure(figsize=(12, 5), facecolor='white')
    plt.subplots_adjust(left=0.2, bottom=0.1, right=0.8, top=0.8, hspace=0.3, wspace=0.15)

    gs = fig.add_gridspec(9, 1)
    ax0 = fig.add_subplot(gs[1, 0], facecolor='white')
    
    ax1 = fig.add_subplot(gs[2, 0], facecolor='white')
    ax2 = fig.add_subplot(gs[3, 0], facecolor='white')
    ax3 = fig.add_subplot(gs[4, 0], facecolor='white')
    ax4 = fig.add_subplot(gs[5, 0], facecolor='white')
    ax5 = fig.add_subplot(gs[6, 0], facecolor='white')

    ax0.set_title("{0}:{1}-{2}".format(chrom, start, end))
       
    plot_bwTrack(ax0, CTCF, "CTCF", chrom, start, end, resolution=resolution , 
                 yminx=0, ymaxx=100,rotation=0, fl='%0.2f',color='#3A6569')
    plot_bwTrack(ax1, POLR2A, "POLR2A", chrom, start, end, resolution=resolution , 
                 yminx=0, ymaxx=100,rotation=0, fl='%0.2f',color='#3A6569')
    plot_bwTrack(ax2, H3K27ac, "H3K27ac", chrom, start, end, resolution=resolution , 
                 yminx=0, ymaxx=100,rotation=0, fl='%0.2f',color='#3A6569')
    plot_bwTrack(ax3, H3K4me3, "H3K4me3", chrom, start, end, resolution=resolution , 
                 yminx=0, ymaxx=100,rotation=0, fl='%0.2f',color='#3A6569')
    plot_bwTrack(ax4, H3K27me3, "H3K27me3", chrom, start, end, resolution=resolution , 
                 yminx=0, ymaxx=100,rotation=0, fl='%0.2f',color='#3A6569')
    plot_bwTrack(ax5, H3K9me3, "H3K9me3", chrom, start, end, resolution=resolution , 
                 yminx=0, ymaxx=100,rotation=0, fl='%0.2f',color='#3A6569')
    fig.savefig("/home/ghaiyan/project/toast/TADResults/GM12878/50kb/{0}:{1}-{2}.pdf".format(chrom, start, end), format='pdf', bbox_inches='tight')
    plt.show()

show("chr19", 50000,20000000, 50000, FDR=0.05)  # 1274