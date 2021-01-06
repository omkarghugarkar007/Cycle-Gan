from tensorflow.keras.layers import Conv2D, LeakyReLU, Input, Activation, Conv2DTranspose,Concatenate
from tensorflow.keras.models import Model,Sequential
from tensorflow_addons.layers import InstanceNormalization
from utils import img_cols,img_rows,channels
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

weight_initialzer = RandomNormal(stddev=0.02)

gen_g_optimizer = gen_f_optimizer = Adam(lr=0.002,beta_1=0.5)
dis_x_optimizer = dis_y_optimizer = Adam(lr=0.002,beta_1=0.5)

def CK(input,filters,use_instancenorm = True):

    block = Conv2D(filters, (4,4),strides=2,padding='same',kernel_initializer=weight_initialzer)(input)

    if use_instancenorm:
        block = InstanceNormalization(axis=-1)(block)
    block = LeakyReLU(0.2)(block)

    return block

def discriminator():

    dis_input = Input(shape=(img_rows,img_cols,channels))

    d = CK(dis_input,64,False)
    d = CK(d,128)
    d = CK(d,256)
    d = CK(d,512)

    d = Conv2D(1,(4,4),padding='same',kernel_initializer=weight_initialzer)(d)

    dis = Model(dis_input,d)
    dis.compile(loss='mse', optimizer=dis_x_optimizer)

    return dis

def DK(filters,use_instancenorm = True):

    block = Sequential()
    block.add(Conv2D(filters,(3,3),strides=2,padding='same',kernel_initializer=weight_initialzer))
    if use_instancenorm:
        block.add(InstanceNormalization(axis=-1))
    block.add(Activation('relu'))

    return block

def UK(filters):

    block = Sequential()
    block.add(Conv2DTranspose(filters,(3,3), strides=2,padding='same',kernel_initializer=weight_initialzer))
    block.add(InstanceNormalization(axis=-1))
    block.add(Activation('relu'))

    return block

def generator():

    gen_input = Input(shape=(img_rows,img_cols,channels))

    enocder_layers = [
        DK(64,False),
        DK(128),
        DK(256),
        DK(512),
        DK(512),
        DK(512),
        DK(512),
        DK(512)
    ]

    deocder_layers = [
        UK(512),
        UK(512),
        UK(512),
        UK(512),
        UK(256),
        UK(128),
        UK(64)
    ]

    gen = gen_input

    skips = []

    for layer in enocder_layers:
        gen = layer(gen)
        skips.append(gen)

    skips = skips[::-1][1:]

    for skip_layer,layer in zip(skips,deocder_layers):

        gen = layer(gen)
        gen = Concatenate()[gen,skip_layer]

    gen = Conv2DTranspose(channels,(3,3),strides=2,padding='same',kernel_initializer=weight_initialzer,activation='tanh')(gen)

    return Model(gen_input,gen)

generator_g = generator()
generator_f = generator()

discriminator_x = discriminator()
discriminator_y = discriminator()

