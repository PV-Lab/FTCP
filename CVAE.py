# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras import  optimizers
from keras.callbacks import ReduceLROnPlateau

from keras.layers import Input,Dense, Lambda,Conv1D,Conv2DTranspose, LeakyReLU,Activation,Flatten,Reshape, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import utils
from utils import *
import featurizer
from featurizer import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

mat_api_key = '5iSAlXJeOTGq30v2'

sup_prop = ['formation_energy_per_atom']
semi_prop = ['Powerfactor']

num_ele=3
num_sites = 20

df, df_in =  get_data(mat_api_key, num_sites+1, 2,num_ele+1)

#num_ele should be equal to most_ele-1
Crystal = crystal_represent(df,num_ele,num_sites)


X=Crystal
    
X= pad(X, 2)


X, scaler_x = minmax(X)


Y = df_in[sup_prop+semi_prop+['ind'] ].values

sup_dim = len(sup_prop )

semi_sup_dim = len(semi_prop)

scaler_y_un = MinMaxScaler()

scaler_y_l = MinMaxScaler()

Y[:,:sup_dim] = scaler_y_un.fit_transform(Y[:,:sup_dim])
Y[:,sup_dim:semi_sup_dim+sup_dim] = scaler_y_l.fit_transform(Y[:,sup_dim:semi_sup_dim+sup_dim])

label_index = df_in.dropna()['ind'].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=21)


#%% VAE
K.clear_session()

input_dim = X_train.shape[1]
channel_dim = X_train.shape[2]
sample_dim = y_train.shape[1]

latent_dim = 256

max_filter = 128

strides = [2,2,1]
kernel = [5,3,3]
#coefficient for KL, regression of ground state properties and regression of sparse properties
reg_kl = 2
reg_sup = 10

reg_semi = 5


x = Input(shape=(input_dim,channel_dim,))


y = Input(shape =(sample_dim ,))

y_label = Input(shape =(semi_sup_dim ,))


label_ind = tf.convert_to_tensor(label_index, dtype =tf.int64)

#y_ind = Input(shape = (1,),dtype=tf.int64)
#y_ind = tf.squeeze(y_ind)

y1 = y[:,:-1]
#y_ind = tf.convert_to_tensor(y_train[:,-1], dtype =tf.int64)
def get_idx(y):
    
    y_ind = y[:,-1]   
    y_ind = tf.dtypes.cast(y_ind,tf.int64)
    com_ind= tf.sets.intersection(y_ind[None,:],label_ind[None,:])
    
    com_ind  = tf.sparse.to_dense(com_ind)
    com_ind = tf.squeeze(com_ind)
    com_ind = tf.reshape(com_ind,(tf.shape(com_ind)[0],1))
    semi_index = tf.where(tf.equal(y_ind,com_ind))[:,-1]
    return semi_index

semi_index = Lambda(get_idx)(y)


def encoder(x):
    
    en0 = Conv1D(max_filter//4,kernel[0],strides= strides[0], padding='SAME')(x)
    en0 = BatchNormalization()(en0)
    en0 = LeakyReLU(0.2)(en0)
    en1 = Conv1D(max_filter//2,kernel[1],strides=strides[1], padding='SAME')(en0)
    en1 = BatchNormalization()(en1)
    en1 = LeakyReLU(0.2)(en1)
    en2 = Conv1D(max_filter,kernel[2], strides=strides[2],padding='SAME')(en1)
#    en2 = MaxPooling1D(2)(en2)
    en2 = BatchNormalization()(en2)
    en2 = LeakyReLU(0.2)(en2)
    en3 = Flatten()(en2)
    en4 = Dense(1024, activation='relu')(en3)
#    en5 = Dense(max_filter,activation = 'sigmoid')(en4)
#    en6= Multiply()([en2,en5])
#    en7 = GlobalAveragePooling1D()(en6)
    
    
    z_mean = Dense(latent_dim,activation = 'linear')(en4)
    z_log_var = Dense(latent_dim,activation = 'linear')(en4)
    
   
    
    return z_mean , z_log_var


z_mean, z_log_var = encoder(x)



epsilon_std =1



def sampling(args):
    z_mean, z_log_var =args
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0],latent_dim),mean=0., stddev = epsilon_std)
    return z_mean+K.exp(0.5*z_log_var/2)*epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean,z_log_var])

encoder_ = Model(x,z)


de0 = Activation('relu')(z_mean)

de1_un = Dense(128,activation = "relu")(de0)

de1_un =Dense(32,activation = "relu")(de1_un)

de1_l = Dense(128,activation = "relu")(de0)

de1_l =Dense(32,activation = "relu")(de1_l)
#
y_predict_sup = Dense(sup_dim, activation ='sigmoid')(de1_un)

y_predict_semi = Dense(semi_sup_dim, activation ='sigmoid')(de1_l)

y_predict_semi = Lambda(lambda x: tf.gather(x,semi_index ,axis=0))(y_predict_semi)

y_label = Lambda(lambda x: tf.gather(x,semi_index ,axis=0))(y)

map_size = K.int_shape(encoder_.layers[-6].output)[1]


z_in = Input(shape=(latent_dim,))


z1 = Dense(max_filter*map_size,activation='relu')(z_in)
z1 = Reshape((map_size,1,max_filter))(z1)
z1 = BatchNormalization()(z1)

#x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
#    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),padding='SAME')(x)

z2 =  Conv2DTranspose( max_filter//2, (kernel[2],1), strides=(strides[2],1),padding='SAME')(z1)
z2 = BatchNormalization()(z2)
z2 = Activation('relu')(z2)



z3 = Conv2DTranspose(max_filter//4, (kernel[1],1), strides=(strides[1],1),padding='SAME')(z2)
z3 = BatchNormalization()(z3)
z3 = Activation('relu')(z3)

z4 = Conv2DTranspose(channel_dim, (kernel[0],1), strides=(strides[0],1),padding='SAME')(z3)
decoded_x = Activation('sigmoid')(z4)

decoded_x = Lambda(lambda x: K.squeeze(x, axis=2))(decoded_x)


decoder_ = Model(z_in,decoded_x)

decoded_x = decoder_(z)


cvae = Model(inputs= [x,y],outputs= decoded_x)

cvae.summary()


def vae_loss(x, decoded_x): 
    
    xent_loss = K.sum(K.square(x[:,:,:]- decoded_x[:,:,:]))

    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
    pred_loss_sup =  K.sum(K.square(y[:,:sup_dim]- y_predict_sup[:,:sup_dim]))
    
    pred_loss_semi = K.sum(K.square( y_predict_semi-y_label[:,sup_dim:semi_sup_dim+sup_dim]))
    vae_loss = K.mean(xent_loss + reg_kl*kl_loss+reg_sup*pred_loss_sup + reg_semi*pred_loss_semi ) 
    
    return vae_loss


reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.3,
                                      patience=4, min_lr=1e-6)


cvae.compile(optimizer = optimizers.rmsprop(lr=8e-4), loss= vae_loss)
#cvae.summary()

cvae.fit([X_train,y_train],X_train,shuffle=True, 
        batch_size=1024,epochs = 500,callbacks=[reduce_lr],
        validation_split=0.0, validation_data=([X_test, y_test], X_test))


train_latent = encoder_.predict(X_train)
test_latent = encoder_.predict(X_test)

plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams.update({'font.size': 16})


real_y_train_un ,real_y_test_un = scaler_y_un.inverse_transform(y_train[:,:sup_dim]),scaler_y_un.inverse_transform(y_test[:,:sup_dim])

#real_y_train_l ,real_y_test_l = scaler_y_l.inverse_transform(y_train[:,unlabel_dim:unlabel_dim+sample_dim_label]),scaler_y_l.inverse_transform(y_test[:,unlabel_dim:unlabel_dim+sample_dim_label])

#plt.scatter(train_latent[:,1],train_latent[:,0],s=7,c=np.squeeze(real_y_train_un[:,0]))
#
#clb = plt.colorbar()
##clb.set_ticklabels(np.unique(sp_train))
#clb.ax.set_title('$E_f$')

len1 = X_train.shape[1]

#%%###################test accuracy ##############################################

vae_x = cvae.predict([X_test, y_test])

plt.scatter(np.linspace(0,len1,len1),vae_x[1,:,2],color ='red',label ='VAE')
plt.scatter(np.linspace(0,len1,len1),X_test[1,:,2],label ='Original')

vae_x = inv_minmax(vae_x,scaler_x)
vae_x [vae_x<0.1] =0
#X_test1 = pad(X_test, 1)

X_test1 = inv_minmax(X_test,scaler_x)

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true+1e-12), np.array(y_pred++1e-12)
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

abc_test = X_test1[:,103,:3]
abc_vae = vae_x[:,103,:3]

ang_test = X_test1[:,104,:3]
ang_vae = vae_x[:,104,:3]

bas_test = X_test1[:,105:105+num_sites,:3]
bas_vae = vae_x[:,105:105+num_sites,:3]

print('abc_PMAE %',round(MAPE(abc_test,abc_vae),3))
print('angle_PMAE %',round(MAPE(ang_test,ang_vae),3))
print('basis_MAE',round(metrics.mean_squared_error(bas_test[:,:,0],bas_vae[:,:,0]),5))
accu = []
vae_ele = []
X_ele = []

Element= joblib.load('./files/element.pkl') 
E_v = np_utils.to_categorical(np.arange(0,len(Element),1))
for i in range(num_ele):
    ele_v = np.argmax(vae_x[:,0:len(E_v),i],axis=1)
    ele_t = np.argmax(X_test1[:,0:len(E_v),i],axis=1)
    vae_ele.append(ele_v)
    X_ele.append(ele_t)
    accu1 =100* round(metrics.accuracy_score(ele_v,ele_t),3)
    accu.append(accu1)
print('Element accuracy %',accu)
