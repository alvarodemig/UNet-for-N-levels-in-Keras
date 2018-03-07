from keras.models import Input, Model
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.layers import Conv2D, concatenate, MaxPooling2D

'''
#####################################################
---------------- UNet STRUCTURE ---------------------
#####################################################
'''

def dbl_conv(prev_layer, channels, activ, do = 0, batch_norm = 0):
    # prev_layer : the previous layer that will be used as input
    # channels: number of channels in the current convolution
    # activ: activation function
    # batch_norm: batch normalization [0 or 1), not included initially
    # do: dropout between convolutions? from 0 to 1
    m = Conv2D(channels, (3, 3), activation=activ, padding='same')(prev_layer)
    if batch_norm != 0: m = BatchNormalization()(m)
    if do > 0 and do < 1: m = Dropout(do)(m)
    m = Conv2D(channels, (3, 3), activation=activ, padding='same')(m)
    if batch_norm != 0: m = BatchNormalization()(m)
    return m
    
def UNet(img_shape, levels=5, initial_channels = 32, channels_rate = 2, activ = 'relu', batch_norm = 0, do = 0):
    # img_shape: shape of input images
    # levels: total number of levels that the UNet will have
    # initial_channels: the number of channels in the 1st level
    # channels_rate: how the number of channels will be modified per level
    # activ: activation function
    # batch_norm: batch normalization [0 or 1), not included initially
    # do: dropout between convolutions? from 0 to 1
    inputs = Input(shape=img_shape)
    channels = initial_channels
    UNet = {'maxpool0': inputs}
    # Level down part of the model
    for i in range(1, levels):
        UNet['conv'+str(i)] = dbl_conv(UNet['maxpool'+str(i-1)], channels, activ, do = do, batch_norm = batch_norm)
        UNet['maxpool'+str(i)] = MaxPooling2D(pool_size=(2, 2))(UNet['conv'+str(i)])
        channels *= channels_rate
    # Lowest Level of the model
    UNet['conv'+str(levels)] = dbl_conv(UNet['maxpool'+str(levels-1)], channels, activ, do = do, batch_norm = batch_norm)
    # Level up part of the model
    for i in range(levels+1, levels*2):
        channels //= channels_rate
        UNet['up'+str(i)] = UpSampling2D(2)(UNet['conv'+str(i-1)])
        UNet['up'+str(i)] = Conv2D(channels, (3, 3), activation=activ, padding='same')(UNet['up'+str(i)])
        UNet['up'+str(i)] = concatenate([UNet['up'+str(i)], UNet['conv'+str(2*levels-i)]], axis=3)
        UNet['conv'+str(i)] = dbl_conv(UNet['up'+str(i)], channels, activ, do = do, batch_norm = batch_norm)
    UNet['conv'+str(2*levels)] = Conv2D(1, (1, 1), activation='sigmoid')(UNet['conv'+str((2*levels)-1)])    
    model = Model(inputs=[inputs], outputs=[UNet['conv'+str(2*levels)] ])
    UNet['inputs'] = UNet.pop('maxpool0')
    return model
