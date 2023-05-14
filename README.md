# TOAST：a method for identifying Tad bOundaries based on graph Auto-encoders and clustering
## 1. before running, you need to install the following package or tools.
Pytorch 1.7.1, Python 3.6.12, Numpy 1.19.5, Matplotlib 3.3.3, Cooltools 0.4.1, Fanc 0.9.24, Jupyter 1.0.0, Pandas 1.1.5, pyBigWig 0.3.18, seaborn 0.11.1.
## 2. prepare data
you can download the simulated Hi-C contact from paper "Forcato, M. et al. Comparison of computational methods for Hi-C data analysis. Nature methods 14, 679-685 (2017)."

you can download the real Hi-C data of GM12878 cell line with GEO ID GSE63525 is provided by the Rao Lab.

For .hic format data, you can dump data using juicer_tools.js and get the NxN matrix, where N is the number of equal-sized regions of a chromosome.

in this paper, we use the NxN format matrix as input.

## 3. Content of folders
1) preprocess:preprocess the Hi-C data.
2) src:call codes for identifying, ploting, and evaluating TADs.
3) TADResilts: all results in this paper.

## 4. Input matrix file format:
The input to CASPIAN is a tab seperated N by N intra-chromosomal contact matrix derived from Hi-C data, where N is the number of equal-sized regions of a chromosome.

## 5. output 
the output is the TAD file at the merge file.
## running
enter the /src folder, you can do the following process:
(1)/src/toast.py: run to get and plot TADs. you can run it by justing changing the input and output folder.

(2)/src/plot_heatmaps.py: plot heatmaps for all TADs

(3)/src/evaulate_TAD.py:evalute TADs

(4)/src/evaulate_TAD-simnarity.py: compare the similarity between TADs.

(5)/src/evaluate-achor-count.py:compute the anchor count for Chip-seq factors.

(6)/src/plot_TAD_markers.py: plot the CHip-seq tracks.

(7)/src/plot-man-figures.py: plot figures in this paper.

