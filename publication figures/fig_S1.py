import joblib
import numpy as np
import matplotlib.pyplot as plt

train_latent, y = joblib.load('./data_files/fig_Eg_latent.joblib')

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

cmap_1 = plt.get_cmap('viridis')
cmap_2 = plt.get_cmap('viridis')

i = 2
j = 11

fig, ax = plt.subplots(1, 2, figsize=(18, 7.3))
s0 = ax[0].scatter(train_latent[:,i],train_latent[:,j],s=7,c=np.squeeze(y.iloc[:,0]), cmap=cmap_1)
cbar = plt.colorbar(s0, ax=ax[0], ticks=list(range(-1, -8, -2)))
ax[0].set_xticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[0].set_yticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[0].set_xlim([-7.5, 7.5])
ax[0].set_ylim([-7.5, 7.5])
s1 = ax[1].scatter(train_latent[:,i],train_latent[:,j],s=7,c=np.squeeze(y.iloc[:,1]), cmap=cmap_2)
plt.colorbar(s1, ax=ax[1], ticks=list(range(0, 10, 2)))
ax[1].set_xticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[1].set_yticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[1].set_xlim([-7.5, 7.5])
ax[1].set_ylim([-7.5, 7.5])
fig.text(0.016, 0.92, '(A) $E_\mathrm{f}$')
fig.text(0.533, 0.92, '(B) $E_\mathrm{g}$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, top=0.85)
plt.show()

fig.savefig('Fig S1.png',dpi=600)