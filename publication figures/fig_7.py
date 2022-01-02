# Only plot partial Figure 7

import joblib
import numpy as np
import matplotlib.pyplot as plt

train_latent, real_y_train_un = joblib.load('./data_files/fig_ICSD_latent.joblib')

from matplotlib import cm
cmap = cm.get_cmap('viridis', 2)

fig = plt.figure()
plt.rcParams["figure.figsize"] = [8.5, 7]
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

i = 2

fig, ax = plt.subplots(1, 2, figsize=(13,5.3))
s0 = ax[0].scatter(train_latent[:,0],train_latent[:,i],s=7,c=np.squeeze(real_y_train_un.iloc[:,1]), cmap=cmap) # real_y_train_un[:,1]
cbar = plt.colorbar(s0, ax=ax[0], ticks=[0.15, 0.85])
cbar.ax.set_yticklabels(['0', '1'])
ax[0].set_xticks([-6,-2,2,6,10])
ax[0].set_yticks([-2, 2, 6, 10, 14])
x, y = 4, 8
ax[0].scatter(x, y, s=150, facecolors='none', edgecolors='#d62728', linewidths=2.5, linestyle='-')
s1 = ax[1].scatter(train_latent[:,0],train_latent[:,i],s=7,c=np.squeeze(real_y_train_un.iloc[:,0])) # real_y_train_un[:,0]
plt.colorbar(s1, ax=ax[1], ticks=[-1, -3, -5, -7])
ax[1].set_xticks([-6,-2,2,6,10])
ax[1].set_yticks([-2, 2, 6, 10, 14])
ax[1].scatter(x, y, s=150, facecolors='none', edgecolors='#d62728', linewidths=2.5, linestyle='-')

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)

fig.text(0.018, 0.97, '(A) ICSD Score')
fig.text(0.510, 0.97, '(B) $E_\mathrm{f}$')