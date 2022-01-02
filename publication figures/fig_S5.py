import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

case_1_Ef_03 = pd.read_excel('./data_files/DFT_Ef_03.xlsx', header=0, engine='openpyxl',)
case_1_Ef_03 = case_1_Ef_03.iloc[:,1]

case_1_Ef_06 = pd.read_excel('./data_files/DFT_Ef_06.xlsx', header=0, engine='openpyxl',)
case_1_Ef_06 = case_1_Ef_06.iloc[:,1]

case_1_Ef_07 = pd.read_excel('./data_files/DFT_Ef_07.xlsx', header=0, engine='openpyxl',)
case_1_Ef_07 = case_1_Ef_07.iloc[:,1]

ind_good = np.squeeze(np.argwhere(np.multiply(case_1_Ef_03>=-0.36,case_1_Ef_03<=-0.24).values))
ind_bad = np.setdiff1d(np.arange(len(case_1_Ef_03)), ind_good)

fig, ax = plt.subplots(1,3, figsize=(18, 7))
ax[0].boxplot(case_1_Ef_03, medianprops=dict(linewidth=2))
ax[0].scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)),case_1_Ef_03[ind_bad])
ax[0].scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)),case_1_Ef_03[ind_good],c='r')
ax[0].set_ylabel('$E_\mathrm{f}$ (eV/atom)')
ax[0].axhline(y=-0.3, color='black', linestyle= '--',label ='$E_\mathrm{f}$ = -0.3 eV/atom')
ax[0].set_xticks([])
ax[0].legend(loc=(0.05,0.8))
ax[0].axhspan(-0.24,-0.36,facecolor='gray',alpha=0.3, zorder=0)

ind_good = np.squeeze(np.argwhere(np.multiply(case_1_Ef_06>=-0.66,case_1_Ef_06<=-0.54).values))
ind_bad = np.setdiff1d(np.arange(len(case_1_Ef_06)), ind_good)

ax[1].boxplot(case_1_Ef_06, medianprops=dict(linewidth=2))
ax[1].scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)),case_1_Ef_06[ind_bad])
ax[1].scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)),case_1_Ef_06[ind_good],c='r')
ax[1].set_ylabel('$E_\mathrm{f}$ (eV/atom)')
ax[1].axhline(y=-0.6, color='black', linestyle= '--',label ='$E_\mathrm{f}$ = -0.6 eV/atom')
ax[1].set_xticks([])
ax[1].legend(loc=(0.05,0.8))
ax[1].axhspan(-0.54,-0.66,facecolor='gray',alpha=0.3, zorder=0)

ind_good = np.squeeze(np.argwhere(np.multiply(case_1_Ef_07>=-0.76,case_1_Ef_07<=-0.64).values))
ind_bad = np.setdiff1d(np.arange(len(case_1_Ef_07)), ind_good)

ax[2].boxplot(case_1_Ef_07, medianprops=dict(linewidth=2))
ax[2].scatter(np.ones(len(ind_bad))+0.04*np.random.normal(0,1,size=len(ind_bad)),case_1_Ef_07[ind_bad])
ax[2].scatter(np.ones(len(ind_good))+0.04*np.random.normal(0,1,size=len(ind_good)),case_1_Ef_07[ind_good],c='r')
ax[2].set_ylabel('$E_\mathrm{f}$ (eV/atom)')
ax[2].axhline(y=-0.7, color='black', linestyle= '--',label ='$E_\mathrm{f}$ = -0.7 eV/atom')
ax[2].set_xticks([])
ax[2].legend(loc=(0.055, 0.2))
ax[2].axhspan(-0.64,-0.76,facecolor='gray',alpha=0.3, zorder=0)

fig.text(0.034, 0.96, '(A)')
fig.text(0.347, 0.96, '(B)')
fig.text(0.678, 0.96, '(C)')

plt.tight_layout()
plt.show()

fig.savefig('Fig S5.png',dpi=600)