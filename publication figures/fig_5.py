import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
plt.rcParams['legend.fontsize'] = font['size']-2.5

mat_api_key = 'YourPymatgenAPI'
mpdr = MPDataRetrieval(mat_api_key)

df_terqua = mpdr.get_dataframe(
    criteria = {
        'nsites': {'$lt': 41},
        'e_above_hull': {'$lt': 0.08},
        'nelements': {'$gt': 2,'$lt': 5},
    },
    properties = [
        'material_id', 
        'formation_energy_per_atom',
        'band_gap', 
        'e_above_hull',
        'pretty_formula',
        'cif',
    ]
)


df_ter = mpdr.get_dataframe(
    criteria = {
        'nsites': {'$lt': 21},
        'e_above_hull':{'$lt':0.08},
        'nelements': {'$gt': 2,'$lt': 4},
    },
    properties = [
        'material_id', 
        'formation_energy_per_atom',
        'band_gap',
        'e_above_hull',
        'pretty_formula',
        'cif',
    ]
)

case_1 = pd.read_excel('./data_files/DFT_Ef_05.xlsx', header=0, engine='openpyxl',)
case_1_Ef = case_1.iloc[:,1]

case_2 = pd.read_excel('./data_files/DFT_Eg.xlsx', header=0, engine='openpyxl',)
case_2 = case_2.iloc[:,:3]
case_2_Ef = case_2.iloc[:,-1].values
case_2_Eg = case_2.iloc[:,-2].values


fig = plt.figure(figsize=(18, 15))
outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)
a = gridspec.GridSpecFromSubplotSpec(1, 5, width_ratios=[8, 0.1, 8, 0.1, 8], 
                                     subplot_spec=outer[0], wspace=0.3)
b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.3)

ind_good = np.squeeze(np.argwhere(np.multiply(case_1_Ef.values>=-0.56,case_1_Ef.values<=-0.44)))
ind_bad = np.setdiff1d(np.arange(len(case_1_Ef)), ind_good)

ax = plt.subplot(a[0])
ax.axhspan(-0.44, -0.56,facecolor='gray',alpha=0.3, zorder=0)
ax.boxplot(case_1_Ef, medianprops=dict(linewidth=2))
ax.scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)),case_1_Ef.iloc[ind_bad],c='#1f77b4')
ax.scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)),case_1_Ef.iloc[ind_good],c='r')
ax.set_ylabel('$E_\mathrm{f}$ (eV/atom)')
ax.axhline(y=-0.5, color='black', linestyle= '--',label ='$E_\mathrm{f}$ = -0.5 eV/atom')
ax.set_xticks([])
ax.set_ylim(-0.6,0.7)
ax.legend(loc=(0.037,0.8), handletextpad=0.4, borderpad=0.2)

ind_good = np.squeeze(np.argwhere(np.multiply(np.multiply(case_2_Eg>=1.2,case_2_Eg<=1.8,),case_2_Ef<-1.44)))
ind_bad = np.setdiff1d(np.arange(len(case_2_Eg)), ind_good)

ax = plt.subplot(a[2])
ax.boxplot(case_2_Eg, medianprops=dict(linewidth=2))
ax.scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)),case_2_Eg[ind_bad])
ax.scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)),case_2_Eg[ind_good], c='r')
ax.set_ylabel('$E_\mathrm{g}$ (eV)')
ax.axhline(y=1.5, color='black', linestyle= '--',label ='$E_\mathrm{g}$ = 1.5 eV')
ax.set_xticks([])
ax.axhspan(1.2,1.8,facecolor='gray',alpha=0.3, zorder=0)
ax.legend(loc=(0.18,0.8), handletextpad=0.4, borderpad=0.2)

ax = plt.subplot(a[4])
ax.boxplot(case_2_Ef, medianprops=dict(linewidth=2))
ax.scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)),case_2_Ef[ind_bad])
ax.scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)),case_2_Ef[ind_good], c='r')
ax.set_ylabel('$E_\mathrm{f}$ (eV/atom)')
ax.axhline(y=-1.5, color='black', linestyle= '--',label ='$E_\mathrm{f}$ < -1.5 eV/atom')
ax.set_ylim(-3,-0.5)
ax.axhspan(-3,-1.44,facecolor='gray',alpha=0.3, zorder=0)
ax.set_xticks([])
ax.legend(loc=(0.037,0.87), handletextpad=0.4, borderpad=0.2)

ax = plt.subplot(b[0])
ax.hist(df_terqua['band_gap'], bins=100,)
ax.set(xlabel='$E_\mathrm{g}$ (eV)', xlim=(-0.2, 6.2))
ax.set_ylabel('Frequency', labelpad=20)
ax.set_yscale('log')
ax.axvspan(1.2,1.8,facecolor='gray',alpha=0.3, zorder=1)
ax.axvline(x=1.5, color='black', linestyle= '--',label ='$E_\mathrm{g}$ = 1.5 eV')
ax.legend()

ax = plt.subplot(b[1])
ax.hist(df_terqua['formation_energy_per_atom'], bins=100, zorder=0)
ax.set(xlabel='$E_\mathrm{f}$ (eV/atom)', xlim=(-5.2,0.2))
ax.set_ylabel('Frequency', labelpad=20)
ax.axvspan(-6, -1.5, facecolor='gray', alpha=0.3, label='$E_\mathrm{f}$ < -1.5 eV/atom', zorder=1)
ax.legend()

ax = fig.add_axes([0.3, 0.44, 0.1, 0.46])
ax.plot([0, 0], [0,1], 'k--')
ax.axis('off')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplots_adjust(top=0.8, bottom=0.1)
fig.text(0.045, 0.85, '(A) Case 1: $E_\mathrm{f}$')
fig.text(0.363, 0.85, '(B) Case 2: $E_\mathrm{g}$ and $E_\mathrm{f}$')
fig.text(0.045, 0.43, '(C)')
fig.text(0.474, 0.41, '(D)')
plt.savefig('Fig 5.png', dpi=600)