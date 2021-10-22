import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Conv1D, Conv2DTranspose, \
    LeakyReLU, Activation, Flatten, Reshape, BatchNormalization
from keras import layers
from keras.models import Model

def FTCP(X_train, y_train, coeffs=(2, 10,), semi=False, label_ind=None, prop_dim=None):
    
    K.clear_session()
    
    if not semi:
        coeff_KL, coeff_prop = coeffs
    else:
        coeff_KL, coeff_prop, coeff_prop_semi = coeffs
    
    latent_dim = 256
    max_filters = 128
    filter_size = [5,3,3]
    strides = [2,2,1]
    
    input_dim = X_train.shape[1]
    channel_dim = X_train.shape[2]
    regression_dim = y_train.shape[1]
    
    encoder_inputs = Input(shape=(input_dim, channel_dim,))
    regression_inputs = Input(shape=(regression_dim,))
    
    if semi:
        assert tuple(label_ind) != None, "You must input the index for semi-supervised property to do semi-supervised learning"
        assert prop_dim != None, "You must input the dimensions of the properties to do semi-supervised learning"
        prop_dim, semi_prop_dim = prop_dim
        
        label_ind = tf.convert_to_tensor(label_ind, dtype=tf.int64)
        def get_idn(y):
            y_ind = y[:,-1]  
            y_ind = tf.dtypes.cast(y_ind, tf.int64)
            com_ind = tf.sets.intersection(y_ind[None, :], label_ind[None, :])
            com_ind = tf.sparse.to_dense(com_ind)
            com_ind = tf.squeeze(com_ind)
            com_ind = tf.reshape(com_ind, (tf.shape(com_ind)[0], 1))
            semi_ind = tf.where(tf.equal(y_ind, com_ind))[:, -1]
            return semi_ind
        semi_ind = Lambda(get_idn)(regression_inputs)
        
    x = Conv1D(max_filters//4, filter_size[0], strides=strides[0], padding='SAME')(encoder_inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(max_filters//2, filter_size[1], strides=strides[1], padding='SAME')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv1D(max_filters, filter_size[2], strides=strides[2], padding='SAME')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dense(1024, activation='sigmoid')(x)
    z_mean = Dense(latent_dim,activation = 'linear')(x)
    z_log_var = Dense(latent_dim,activation = 'linear')(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean+K.exp(z_log_var/2)*epsilon
    
    # Reparameterization
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(encoder_inputs, z, name='encoder')
    
    if not semi:
        x = Activation('relu')(z_mean)
        x = Dense(128, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        y_hat = Dense(regression_dim, activation ='sigmoid')(x)
        regression = Model(encoder_inputs, y_hat, name='target-learning branch')
    else:
        x = Activation('relu')(z_mean)
        x = Dense(128, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        y_hat = Dense(prop_dim, activation ='sigmoid')(x)
        
        x = Activation('relu')(z_mean)
        x = Dense(128, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        y_semi_hat = Dense(semi_prop_dim, activation ='sigmoid')(x)
        regression = Model(encoder_inputs, [y_hat, y_semi_hat], name='target-learning branch')
        
        y_semi = Lambda(lambda x: tf.gather(x, semi_ind, axis=0))(regression_inputs)
        y_semi_hat = Lambda(lambda x: tf.gather(x, semi_ind, axis=0))(y_semi_hat)
    
    latent_inputs = Input(shape=(latent_dim,))
    map_size = K.int_shape(encoder.layers[-6].output)[1]
    x = Dense(max_filters*map_size, activation='relu')(latent_inputs)
    x = Reshape((map_size, 1, max_filters))(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(max_filters//2, (filter_size[2], 1), strides=(strides[2], 1), 
                        padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(max_filters//4, (filter_size[1], 1), strides=(strides[1], 1), 
                        padding='SAME')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(channel_dim, (filter_size[0],1), strides=(strides[0],1), 
                        padding='SAME')(x)
    x = Activation('sigmoid')(x)
    decoder_outputs = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    
    reconstructed_outputs = decoder(z)
    VAE = Model(inputs=[encoder_inputs, regression_inputs], outputs=reconstructed_outputs)
    
    VAE.summary()
    
    def vae_loss(x, decoded_x):
        loss_recon = K.sum(K.square(encoder_inputs - reconstructed_outputs))
        loss_KL = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        loss_prop = K.sum(K.square(regression_inputs[:, :prop_dim] - y_hat))
        
        if semi:
            loss_prop_semi = K.sum(K.square(y_semi_hat - y_semi[:, prop_dim:prop_dim+semi_prop_dim]))
            vae_loss = K.mean(loss_recon + coeff_KL*loss_KL + coeff_prop*loss_prop + coeff_prop_semi*loss_prop_semi)
        else:
            vae_loss = K.mean(loss_recon + coeff_KL*loss_KL + coeff_prop*loss_prop)
        return vae_loss
    
    return VAE, encoder, decoder, regression, vae_loss