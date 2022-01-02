from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from tqdm import tqdm
from pymatgen import Structure
import joblib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [6, 7]
font = {
        'family': 'Avenir',
        'weight': 'normal',
        'size': 26
    }
math_font = 'stixsans'
plt.rc('font', **font)
plt.rcParams['mathtext.fontset'] = math_font
plt.rcParams['axes.labelsize'] = font['size']
plt.rcParams['xtick.labelsize'] = font['size']-2
plt.rcParams['ytick.labelsize'] = font['size']-2
plt.rcParams['legend.fontsize'] = font['size']-2

ssf = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0), 
    stats=('mean', 'std_dev', 'minimum', 'maximum'))
v_new =[]
name = np.arange(29)
for folder in ('../FTCP-designed compounds/Case 1/Ef_03/', '../FTCP-designed compounds/Case 1/Ef_05/', 
               '../FTCP-designed compounds/Case 1/Ef_06/', '../FTCP-designed compounds/Case 1/Ef_07/'):
    for j in tqdm(name):
        try:
            new_crystal = Structure.from_file(f"{folder}{j}_fin.cif")
            v_new.append(np.array(ssf.featurize(new_crystal)))
        except FileNotFoundError:
            try:
                new_crystal = Structure.from_file(f"{folder}gen{j}_fin.cif")
                v_new.append(np.array(ssf.featurize(new_crystal)))
            except FileNotFoundError:
                pass
v_new = np.array(v_new)

v_database = joblib.load('./data_files/ter_20_lt_0.08.joblib')
idx = joblib.load('./data_files/ter_20_lt_0.08_idx.joblib')

from sklearn.metrics import pairwise_distances
case_1 = pairwise_distances(v_new, v_database)
case_1 = np.min(case_1, axis=1)
print(np.median(case_1))
print(len(case_1[case_1>=0.75]))

name = list(range(20))
ssf = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
    stats=('mean', 'std_dev', 'minimum', 'maximum'))
v_new =[]
for j in tqdm(name):
    try:
        new_crystal = Structure.from_file(f"../FTCP-designed compounds/Case 2/gen{j}_fin.cif")
        v_new.append(np.array(ssf.featurize(new_crystal)))
    except FileNotFoundError:
        pass
v_new = np.array(v_new)

v_database = joblib.load('./data_files/terqua_40_lt_0.08.joblib')
idx = joblib.load('./data_files/terqua_40_lt_0.08_idx.joblib')

case_2 = pairwise_distances(v_new, v_database)
case_2 = np.min(case_2, axis=1)
print(np.median(case_2))
print(len(case_2[case_2>=0.75]))

name = list(range(28))

ssf = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
    stats=('mean', 'std_dev', 'minimum', 'maximum'))
v_new =[]
for j in tqdm(name):
    try:
        new_crystal = Structure.from_file(f"../FTCP-designed compounds/Case 3/CONTCAR-gen{j}-sym.cif")
        v_new.append(np.array(ssf.featurize(new_crystal)))
    except FileNotFoundError:
        pass
v_new = np.array(v_new)

v_database = joblib.load('./data_files/terqua_40_lt_0.08.joblib')
idx = joblib.load('./data_files/terqua_40_lt_0.08_idx.joblib')

case_3 = pairwise_distances(v_new, v_database)
case_3 = np.min(case_3, axis=1)
print(np.median(case_3))
print(len(case_3[case_3>=0.75]))

ind_good = [6, 8, 9, 19, 20, 22, 23, 27, 28, 30, 35, 
            43, 48, 49, 50, 61, 68, 69, 71, 74, 81]
ind_bad = np.setdiff1d(np.arange(len(case_1)), ind_good)

fig, ax = plt.subplots(1, 3, figsize=(18,7))
ax[0].boxplot(case_1, medianprops=dict(linewidth=2))
ax[0].scatter(np.ones(len(ind_bad))+0.05*np.random.normal(0,1,size=len(ind_bad)), case_1[ind_bad],
            s=30, alpha=0.8)
ax[0].scatter(np.ones(len(ind_good))+0.05*np.random.normal(0,1,size=len(ind_good)), case_1[ind_good],
            s=30, alpha=0.8, c='r')
ax[0].set_ylabel('Dissimilarity')
ax[0].set_xticks([])
ax[0].set_ylim(0, 1.3)

ind_good = [2, 10, 11, 12, 13, 14, 15]
ind_bad = np.setdiff1d(np.arange(len(case_2)), ind_good)

ax[1].boxplot(case_2, medianprops=dict(linewidth=2))
ax[1].scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)), case_2[ind_bad],
            s=30, alpha=0.8)
ax[1].scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)), case_2[ind_good],
            s=30, alpha=0.8, c='r')
ax[1].set_ylabel('Dissimilarity')
ax[1].set_xticks([])
ax[1].set_ylim(0, 1.3)

ind_good = [0, 7]
ind_bad = np.setdiff1d(np.arange(len(case_3)), ind_good)

ax[2].boxplot(case_3, medianprops=dict(linewidth=2))
ax[2].scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)), case_3[ind_bad],
            s=30, alpha=0.8)
ax[2].scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)), case_3[ind_good],
            s=30, alpha=0.8, c='r')
ax[2].set_ylabel('Dissimilarity')
ax[2].set_xticks([])
ax[2].set_ylim(0,1.3)

fig.text(0.021, 0.96, '(A) Case 1')
fig.text(0.3565, 0.96, '(B) Case 2')
fig.text(0.689, 0.96, '(C) Case 3')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5, top=0.9)

fig.savefig('Fig S6.png', dpi=600)