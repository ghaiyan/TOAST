#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statannot import add_stat_annotation

import matplotlib as mpl

mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
sns.set(style="ticks")
"""
name = 'Toast-tad-quality'
dir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb"
dat = pd.read_excel(dir+"/tadSimilarity.xlsx", sheet_name=name)
order=['Toast','Caspian', 'TopDom', 'IS','HiCseg','IC_Finder','clusterTAD',"DI"]
plt.figure(figsize=(8,4))
line = sns.boxplot(data = dat,  whis=2.5)
sns.stripplot(data = dat)

line, test_results = add_stat_annotation(line, data=dat, order=order,
                                   box_pairs=[("Toast", "Caspian"),("Toast", "TopDom"),("Toast", "IS"),("Toast", "HiCseg"), ("Toast", "IC_Finder"), ("Toast", "clusterTAD"), ("Toast", "DI")],
                                   test='Mann-Whitney', text_format='full', loc='inside', verbose=2)
plt.savefig(dir+'/'+name+'-similarity.pdf',pad_inches=0,bbox_inches='tight')
plt.show()
"""

"""
dir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb"
dat = pd.read_excel(dir+"/tadSimilarity.xlsx", sheet_name=name)
order=['Caspian', 'TopDom', 'IS','HiCseg','IC_Finder','clusterTAD',"DI"]
plt.figure(figsize=(8,4))
line = sns.boxplot(data = dat,  whis=2.5)
sns.stripplot(data = dat)

line, test_results = add_stat_annotation(line, data=dat, order=order,
                                   box_pairs=[("Caspian", "TopDom"),("Caspian", "IS"),("Caspian", "HiCseg"), ("Caspian", "IC_Finder"), ("Caspian", "clusterTAD"), ("Caspian", "DI")],
                                   test='Mann-Whitney', text_format='full', loc='inside', verbose=2)
plt.savefig(dir+'/'+name+'-similarity.pdf',pad_inches=0,bbox_inches='tight')
plt.show()
"""
"""
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import numpy as np
dir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb"
metricsdir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb/compareWithraw.txt"
metricsFileName='compareWithraw'

RI=np.loadtxt(metricsdir,usecols=(0,))
FMS=np.loadtxt(metricsdir,usecols=(1,))

x=np.arange(1,23,1)
l1=plt.plot(x,RI,'r--',label='RI')
l2=plt.plot(x,FMS,'g--',label='FMS')

plt.plot(x,RI,'ro-',x,FMS,'g+-')

plt.ylabel('metric value')
plt.xlabel('methods')
plt.legend()
plt.savefig(dir+'/'+metricsFileName+'_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import numpy as np
dir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb"
metricsdir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb/compareWithraw.txt"
metricsFileName='compareWithraw-tad-quality'

kr_tad_num=np.loadtxt(metricsdir,usecols=(4,))
raw_tad_num=np.loadtxt(metricsdir,usecols=(5,))

x=np.arange(1,23,1)
l1=plt.plot(x,kr_tad_num,'r--',label='kr_tad_quality')
l2=plt.plot(x,raw_tad_num,'g--',label='raw_tad_quality')

plt.plot(x,kr_tad_num,'ro-',x,raw_tad_num,'g+-')

plt.ylabel('metric value')
plt.xlabel('methods')
plt.legend()
plt.savefig(dir+'/'+metricsFileName+'_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()


"""
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
import numpy as np
dir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb"
metricsdir="/home/ghaiyan/project/toast/TADResults/GM12878/lr=0.001-hdbscan/50kb/different-resolution-compare.txt"
metricsFileName='different-resolution-compare-tad-quality'

_50kb=np.loadtxt(metricsdir,usecols=(3,))
_25kb=np.loadtxt(metricsdir,usecols=(4,))
_5kb=np.loadtxt(metricsdir,usecols=(5,))

x=np.arange(1,20,1)
l1=plt.plot(x,_50kb,'r--',label='_50kb')
l2=plt.plot(x,_25kb,'g--',label='_25kb')
l3=plt.plot(x,_5kb,'b--',label='_5kb')

plt.plot(x,_50kb,'ro-',x,_25kb,'g+-',x,_5kb,'b^-')

plt.ylabel('TAD quality')
plt.xlabel('chromosome number')
plt.legend()
plt.savefig(dir+'/'+metricsFileName+'_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()
"""