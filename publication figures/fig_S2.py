import joblib
import numpy as np
from tqdm import tqdm

train_latent, y = joblib.load('./data_files/fig_Ef_latent.joblib')

ef = -0.5
num = 2

def find_nearest(array,target,num, a=1):
    array = np.sum(np.abs(array-target),axis=1)
    idx = np.argsort(array)
    return idx[:num*a:a]

import itertools

def get_slerp(inv_train, aug_num):
    inv_train_s = []
    
    def slerp(v0, v1, t_array):
        """Spherical linear interpolation."""
        # >>> slerp([1,0,0,0], [0,0,0,1], np.arange(0, 1, 0.001))
        t_array = np.array(t_array)
        v0 = np.array(v0)
        v1 = np.array(v1)
        dot = np.sum(v0 * v1)/np.sqrt(np.sum(np.square(v0)))/np.sqrt(np.sum(np.square(v1)))
    
        if dot < 0.0:
            v1 = -v1
            dot = -dot
        
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            result = v0[np.newaxis,:] + t_array[:,np.newaxis] * (v1 - v0)[np.newaxis,:]
            return (result.T / np.linalg.norm(result, axis=1)).T
        
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
    
        theta = theta_0 * t_array
        sin_theta = np.sin(theta)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0[:,np.newaxis] * v0[np.newaxis,:]) + (s1[:,np.newaxis] * v1[np.newaxis,:])
    
    for a in tqdm(itertools.combinations(inv_train,2)):
        inv_train_s.append(slerp(a[0],a[1],np.linspace(0,1,aug_num)))   
    return  np.vstack(inv_train_s)

idx = find_nearest(y,ef,num, a=5)
inv_train = train_latent[idx,:]
slerp = get_slerp(inv_train,10)

idx = find_nearest(y,ef,num, a=2)
inv_train = train_latent[idx,:]

aug_num = 20
inv_train = np.tile(inv_train[:1],(aug_num,1))
noise_vec = np.random.normal(0,1,inv_train.shape)
local_perturb = inv_train + noise_vec*0.4

aug_num = 250
latent_mean = np.mean(train_latent, axis=0)
latent_std = np.std(train_latent, axis=0)
gaussian = np.random.normal(latent_mean,latent_std,(aug_num, train_latent.shape[1]))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('viridis')
new_cmap = truncate_colormap(cmap, 0.3, 1)

i = 23
j = 202

fig = plt.figure(figsize=(18, 6))
outer = gridspec.GridSpec(1, 1, figure=fig,)
a = gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[4,4,5], 
                                     subplot_spec=outer[0], wspace=0.25)
ax = []
ax.append(plt.subplot(a[0]))
ax.append(plt.subplot(a[1]))
ax.append(plt.subplot(a[2]))

# fig, ax = plt.subplots(1, 3, figsize=(18, 6))
s0 = ax[0].scatter(train_latent[:,i],train_latent[:,j],s=7,c=np.squeeze(y.iloc[:,0]), cmap=new_cmap)
ax[0].set_xticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[0].set_yticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[0].scatter(local_perturb[:,i], local_perturb[:,j],s=80, edgecolors='r',facecolors='none',)
ax[0].set_xlim(-7.7, 7.7)
ax[0].set_ylim(-7.7, 7.7)

s1 = ax[1].scatter(train_latent[:,i],train_latent[:,j],s=7,c=np.squeeze(y.iloc[:,0]), cmap=new_cmap)
ax[1].set_xticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[1].set_yticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[1].scatter(slerp[:,i], slerp[:,j],s=100, edgecolors='r',facecolors='none',)
ax[1].set_xlim(-7.7, 7.7)
ax[1].set_ylim(-7.7, 7.7)

s2 = ax[2].scatter(train_latent[:,i],train_latent[:,j],s=7,c=np.squeeze(y.iloc[:,0]), cmap=new_cmap)
ax[2].set_xticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[2].set_yticks([-7, -5, -3, -1, 1, 3, 5, 7])
ax[2].scatter(gaussian[:,i], gaussian[:,j],s=100, edgecolors='r',facecolors='none',)
ax[2].set_xlim(-7.7, 7.7)
ax[2].set_ylim(-7.7, 7.7)

cbar = plt.colorbar(s2, ax=ax[2], ticks=list(range(0, -7, -1)))

fig.text(0.016, 0.95, '(A) Lp')
fig.text(0.317, 0.95, '(B) Slerp')
fig.text(0.624, 0.95, '(C) Gp')

fig.text(0.93, 0.95, '$E_\mathrm{f}$')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, top=0.85)
plt.show()

fig.savefig('Fig S2.png',dpi=600)