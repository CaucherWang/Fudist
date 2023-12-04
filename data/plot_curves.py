# encoding=utf-8
import os
import numpy as np
from pprint import pprint
from utils import *

fig = plt.figure(figsize=(6,5))

x = np.arange(0, 105, 5)
deep = [57.0, 59.0, 51.0, 41.0, 35.0, 30.0, 26.0, 24.0, 24.0, 23.0, 23.0, 21.0, 21.0, 21.0, 21.0, 20.0, 19.0, 18.0, 19.0, 15.0, 5.0]
sift = [95.0, 71.0, 38.0, 22.5, 17.0, 16.0, 17.0, 19.0, 20.0, 22.0, 23.0, 24.0, 25.0, 27.0, 30.0, 35.0, 40.0, 43.0, 37.0, 36.0, 53.0]
word2vec = [28.0, 9.0, 3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0, 0.0]
glove = [14609.0, 2660.0, 389.0, 50.0, 7.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

x = [0.01, 0.1, 1, 5, 10]

y = {
    'deep':{
        'NSW':np.array([0.75, 3.05, 9.95, 21.84, 30.56]),
        'pNSW':np.array([0.0240625, 0.0705, 0.155903125, 0.303508125, 0.4072484375]) * 100,
        '$k$-Graph':np.array([6.88, 18.08, 34.05, 52.72, 63]),
        'HNSW$_0$':np.array([0.005050505050505051, 0.010528324692261204, 0.040897530022549686, 0.11576005385600141, 0.18849391273179072])* 100,
        'NSG':np.array([0.0125, 0.02740625, 0.075115625, 0.17131125, 0.2576528125])* 100,
        'random':x
    },
    'sift':{
        'NSW':np.array([1.09, 4.13, 11.28, 23.14, 31.77]),
        'pNSW':np.array([0.0534375, 0.08971875, 0.171471875, 0.312724375, 0.4106284375])* 100,
        '$k$-Graph':np.array([5.88, 13.73, 30.03, 48.67, 59.39]),
        'HNSW$_0$':np.array([0.010995915802701853, 0.01555928022516682, 0.051295841436099915, 0.13348493980125303, 0.20740160070247177])* 100,
        'NSG':np.array([0.01875, 0.03190625, 0.08284739279620498, 0.18788372748574783, 0.2781263156118175])* 100,
        'random':x
    },
    'glove1.2m':{
        'NSW':np.array([0.0004965355979634606, 0.0030605757964386046, 0.022611954995797507, 0.09702755526128912, 0.18544996549507162])* 100,
        'pNSW':np.array([0.38918067226890757, 0.4736221709974853, 0.5310745705906996, 0.5892354419773774, 0.6569003925396519])* 100,
        '$k$-Graph':np.array([0.5265231092436975, 0.8135477787091366, 0.8534326560536238, 0.8910515291160452, 0.9234476983854346])* 100,
        'HNSW$_0$':np.array([0.06311323256430655, 0.09881349647756767, 0.15171552663411003, 0.4995315299162971, 0.7062133475973404])* 100,
        'NSG':np.array([0.0009097365285627923, 0.00904343252781577, 0.1135128430970715, 0.40448713548225573, 0.548466980169226])* 100,
        'random':x
    },
    'word2vec':{
        'NSW':np.array([0.0006834235015837419, 0.003951412361580565, 0.02719426594305959, 0.10476625184541012, 0.19137333328094652])* 100,
        'pNSW':np.array([0.1990625, 0.42896875, 0.487659375, 0.52190875, 0.6059515625])* 100,
        '$k$-Graph':np.array([0.3540625, 0.60109375, 0.75051875, 0.842195625, 0.8869690625])* 100,
        'HNSW$_0$':np.array([0.020491803278688523, 0.04822294790907073, 0.09919541179894997, 0.37157191112124627, 0.5156563595084616])* 100,
        'NSG':np.array([0.007358116322221511, 0.032996691624586454, 0.12289788682497405, 0.3432835351239431, 0.47263858124914876])* 100,
        'random':x
    },
    'rand200':{
        'NSW':np.array([0.00034350578885323715, 0.003918968253376944, 0.031425356858872065, 0.12128494677660255, 0.20961187498151926])* 100,
        'pNSW':np.array([0.0071875, 0.0329375, 0.1234125, 0.288781875, 0.40459875])* 100,
        '$k$-Graph':np.array([0.0128125, 0.05503125, 0.20509375, 0.43895375, 0.5811153125])* 100,
        'HNSW$_0$':np.array([0.005, 0.03440625, 0.143125, 0.3424084921486126, 0.4675304544089396])* 100,
        # 'HNSW$_0$':np.array([0.006875, 0.03025, 0.11912849727655399, 0.2728469329399963, 0.38446848245525217])* 100, efcons: 10000
        'NSG':np.array([0.010906258114775382, 0.04922887468480619, 0.2189292601736866, 0.5089827954889845, 0.6472834663336051])* 100,
        'random':x
    },
    'gauss200':{
        'NSW':np.array([0.0005002641145946618, 0.003851252229225944, 0.02759240825709919, 0.10899231607416127, 0.19475703093171928])* 100,
        'pNSW':np.array([0.025625, 0.102375, 0.264803125, 0.46480625, 0.557565625])* 100,
        '$k$-Graph':np.array([0.051875, 0.177875, 0.46249375, 0.7311575, 0.83928125])* 100,
        'HNSW$_0$':np.array([0.028125, 0.0910625, 0.26248914055712846, 0.4889141817095405, 0.5675622170619004])* 100,
        'NSG':np.array([0.009960431163869559, 0.07718719544193718, 0.3691120048074701, 0.6773570189148426, 0.7566446388605564])* 100,
        'random':x
    },
}
    


k = 20
width = 2


dataset = 'gauss200'



plt.plot(x, y[dataset]['$k$-Graph'], **plot_info[0], label='$k$-Graph')
plt.plot(x, y[dataset]['NSW'], **plot_info[1], label='NSW')
plt.plot(x, y[dataset]['pNSW'], **plot_info[2], label='pNSW')
plt.plot(x, y[dataset]['HNSW$_0$'], **plot_info[3], label='HNSW$_0$')
plt.plot(x, y[dataset]['NSG'], **plot_info[4], label='NSG')
plt.plot(x, y[dataset]['random'], **plot_info[5], label='random')


# plt.legend(loc="best", fontsize=15)
# plt.xticks()
# plt.yticks()
# plt.xlabel("Percentage distance to mean (%)")
# plt.ylabel(r"Median of $k$-occurence")

plt.ylabel("Out-link density (%)")
plt.xlabel("#points with the largest in-degree (%)")
# plt.legend(loc='best')

# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.)  #显示图中左上角的标识区域
# plt.title('rand-256-100m (100GB)', fontsize=20)
# plt.legend(loc='best',ncol = 2, fontsize=16)  #显示图中左上角的标识区域
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=6,mode="expand", borderaxespad=0.,fontsize=15.5)  #显示图中左上角的标识区域

plt.tight_layout()
# plt.savefig(f'./figures/bulletin-{dataset}-dist2mean.png',  format='png')
plt.savefig(f'./figures/{dataset}-outlink-density.png',  format='png')
# plt.show()
# plt.savefig('../figs/approx-node-full-%s-recall.png' % ds,  format='png')
# plt.savefig('../figs/approx-full-title.png', format='png')
