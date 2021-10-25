from data import *
from model import *
from utils import *
from sampling import *

import joblib
import numpy as np
import matplotlib.pyplot as plt

from keras import  optimizers
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Query ternary and quaternary compounds with number of sites <= 40
max_elms = 4
min_elms = 3
max_sites = 40
# Use your own API key to query Materials Project (https://materialsproject.org/open)
mp_api_key = 'YourAPIKey'
dataframe = data_query(mp_api_key, max_elms, min_elms, max_sites, include_te=True)

# Obtain FTCP representation
FTCP_representation, Nsites = FTCP_represent(dataframe, max_elms, max_sites, return_Nsites=True)
# Preprocess FTCP representation to obtain input X
FTCP_representation = pad(FTCP_representation, 2)
X, scaler_X = minmax(FTCP_representation)

# Get Y from queried dataframe
prop = ['formation_energy_per_atom', 'band_gap', 'Powerfactor', 'ind']

prop_dim = 2
semi_prop_dim = 1

Y = dataframe[prop].values
scaler_y = MinMaxScaler()
scaler_y_semi = MinMaxScaler()
Y[:, :prop_dim] = scaler_y.fit_transform(Y[:, :prop_dim])
Y[:, prop_dim:prop_dim+semi_prop_dim] = scaler_y_semi.fit_transform(Y[:, prop_dim:prop_dim+semi_prop_dim])

# Get training, and test data; feel free to have a validation set if you need to tune the hyperparameter
ind_train, ind_test = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=21)
X_train, X_test = X[ind_train], X[ind_test]
y_train, y_test = Y[ind_train], Y[ind_test]

# Get model
VAE, encoder, decoder, regression, vae_loss = FTCP(X_train, 
                                                   y_train, 
                                                   coeffs=(3, 20, 5,), 
                                                   semi=True,
                                                   label_ind=dataframe.dropna()['ind'].values,
                                                   prop_dim=(prop_dim, semi_prop_dim),
                                                   )
# Train model
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=4, min_lr=1e-6)
def scheduler(epoch, lr):
    if epoch == 50:
        lr = 2e-4
    elif epoch == 100:
        lr = 5e-5
    return lr
schedule_lr = LearningRateScheduler(scheduler)

VAE.compile(optimizer=optimizers.rmsprop(lr=8e-4), loss=vae_loss)
VAE.fit([X_train, y_train], 
        X_train,
        shuffle=True, 
        batch_size=256,
        epochs=200,
        callbacks=[reduce_lr, schedule_lr],
        )

#%% Visualize latent space with two arbitrary dimensions
train_latent = encoder.predict(X_train, verbose=1)
y_train_ = np.concatenate((scaler_y.inverse_transform(y_train[:, :prop_dim]), 
                           scaler_y_semi.inverse_transform(y_train[:, prop_dim:prop_dim+semi_prop_dim])),
                          axis=1
                          )
y_test_ = np.concatenate((scaler_y.inverse_transform(y_test[:, :prop_dim]), 
                           scaler_y_semi.inverse_transform(y_test[:, prop_dim:prop_dim+semi_prop_dim])),
                          axis=1
                          ) 

font_size = 26
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size-2
plt.rcParams['ytick.labelsize'] = font_size-2

fig, ax = plt.subplots(1, 3, figsize=(18, 5.3))
s0 = ax[0].scatter(train_latent[:,0], train_latent[:,1], s=7, c=np.squeeze(y_train_[:,0]))
plt.colorbar(s0, ax=ax[0], ticks=list(range(-1, -8, -2)))
s1 = ax[1].scatter(train_latent[:,0], train_latent[:,1], s=7, c=np.squeeze(y_train_[:,1]))
plt.colorbar(s1, ax=ax[1], ticks=list(range(0, 10, 2)))
s2 = ax[2].scatter(train_latent[:,0], train_latent[:,1], s=7, c=np.squeeze(y_train_[:,2]))
plt.colorbar(s2, ax=ax[2])
fig.text(0, 0.92, '(A) $E_\mathrm{f}$', fontsize=font_size)
fig.text(0.33, 0.92, '(B) $E_\mathrm{g}$', fontsize=font_size)
fig.text(0.678, 0.92, '(C) Power Factor', fontsize=font_size)

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, top=0.85)
plt.show()

#%% Evalute Reconstruction, and Target-Learning Branch Error
X_test_recon = VAE.predict([X_test, y_test], verbose=1)
X_test_recon_ = inv_minmax(X_test_recon, scaler_X)
X_test_recon_[X_test_recon_ < 0.1] = 0
X_test_ = inv_minmax(X_test, scaler_X)

# Mean absolute percentage error
def MAPE(y_true, y_pred):
    # Add a small value to avoid division of zero
    y_true, y_pred = np.array(y_true+1e-12), np.array(y_pred+1e-12)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Mean absolute error
def MAE(y_true, y_pred):
    return np.nanmean(np.abs(y_true - y_pred), axis=0)

# Mean absolute error for reconstructed site coordinate matrix
def MAE_site_coor(SITE_COOR, SITE_COOR_recon, Nsites):
    site = []
    site_recon = []
    # Only consider valid sites, namely to exclude zero padded (null) sites
    for i in range(len(SITE_COOR)):
        site.append(SITE_COOR[i, :Nsites[i], :])
        site_recon.append(SITE_COOR_recon[i, :Nsites[i], :])
    site = np.vstack(site)
    site_recon = np.vstack(site_recon)
    return np.mean(np.ravel(np.abs(site - site_recon)))

# Read string of elements considered in the study (to get dimension for element matrix)
elm_str = joblib.load('data/element.pkl')
# Get lattice constants, abc
abc = X_test_[:, len(elm_str), :3]
abc_recon = X_test_recon_[:, len(elm_str), :3]
print('abc (MAPE): ', MAPE(abc,abc_recon))

# Get lattice angles, alpha, beta, and gamma
ang = X_test_[:, len(elm_str)+1, :3]
ang_recon = X_test_recon_[:, len(elm_str)+1, :3]
print('angles (MAPE): ', MAPE(ang, ang_recon))

# Get site coordinates
coor = X_test_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
coor_recon = X_test_recon_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
print('coordinates (MAE): ', MAE_site_coor(coor, coor_recon, Nsites[ind_test]))

# Get accuracy of reconstructed elements
elm_accu = []
for i in range(max_elms):
    elm = np.argmax(X_test_[:, :len(elm_str), i], axis=1)
    elm_recon = np.argmax(X_test_recon_[:, :len(elm_str), i], axis=1)
    elm_accu.append(metrics.accuracy_score(elm, elm_recon))
print(f'Accuracy for {len(elm_str)} elements are respectively: {elm_accu}')

# Get target-learning branch regression error
y_test_hat = regression.predict(X_test, verbose=1)
y_test_hat_ = scaler_y.inverse_transform(y_test_hat[0])
y_test_semi_hat_ = scaler_y_semi.inverse_transform(y_test_hat[1])
print(f'The regression MAE for {prop[:prop_dim]} are respectively', MAE(y_test_[:, :prop_dim], y_test_hat_))
print(f'The regression MAE for {prop[prop_dim:prop_dim+semi_prop_dim]} are respectively', MAE(y_test_[:, prop_dim:prop_dim+semi_prop_dim], y_test_semi_hat_))

#%% Sampling the latent space and perform inverse design

# Specify design targets, 0.3 eV <= Eg <= 1.5 eV, Ef < 0 eV/atom (power factor as high as possible)
target_Ef, target_Eg_min, target_Eg_max = -1.5, 0.3, 1.5
# Set number of compounds to purturb locally about
Nsamples = 10
# Obtain points that are closest to the design target in the training set
ind_constraint_1 = np.squeeze(np.argwhere(y_train_[:, 0] < target_Ef))
ind_constraint_2 = np.squeeze(np.argwhere(y_train_[:, 1] >= target_Eg_min))
ind_constraint_3 = np.squeeze(np.argwhere(y_train_[:, 1] <= target_Eg_max))
ind_constraint = np.intersect1d(np.intersect1d(ind_constraint_1, ind_constraint_2), ind_constraint_3)
# Sort the latent space according to the value of predicted power factor
y_train_semi_hat = regression.predict(X_train, verbose=1)[1]
ind_temp = np.argsort(-y_train_semi_hat[ind_constraint, 0])
ind_sample = ind_constraint[ind_temp][:Nsamples]
# Set number of purturbing instances around each compound
Nperturb = 3
# Set local purturbation (Lp) scale
Lp_scale = 0.9

# Sample (Lp)
samples = train_latent[ind_sample, :]
samples = np.tile(samples, (Nperturb, 1))
gaussian_noise = np.random.normal(0, 1, samples.shape)
samples = samples + gaussian_noise * Lp_scale
ftcp_designs = decoder.predict(samples, verbose=1)
ftcp_designs = inv_minmax(ftcp_designs, scaler_X)

# Get chemical info for designed crystals and output CIFs
pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, ind_unique = get_info(ftcp_designs, 
                                                                                   max_elms, 
                                                                                   max_sites, 
                                                                                   elm_str=joblib.load('data/element.pkl'),
                                                                                   to_CIF=True,
                                                                                   check_uniqueness=True,
                                                                                   mp_api_key=mp_api_key,
                                                                                   )