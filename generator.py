import tensorflow as tf
from resnet_block import ResNetBlockInstanceNorm
from suppixel import SubpixelConv2D
import tensorflow_addons as tfa
# import tensorlayer as tfl
# from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)

# from tensorlayer.layers import SubpixelConv2d


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        layers = tf.keras.layers
        # print("aa")
        # self.layers["a"] = Layers.Conv2D(64,(1,1),padding = 'same',kernel_initializer=init )
        self.layers.append( layers.Conv2D(32,(3,3),padding = 'same',kernel_initializer=init ))
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2D(64,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2D(128,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        for _ in range(4):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=128) )
        self.layers.append( layers.Conv2DTranspose(128,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2DTranspose(64,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2D(3,(1,1), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( tf.keras.activations.tanh )

    def call(self,input_image):
        # print(input_image.shape)
        for layer in self.layers:
            input_image = layer(input_image)
            # print(input_image.shape)

        return input_image

class GeneratorV2(tf.keras.layers.Layer):
    def __init__(self):
        super(GeneratorV2,self).__init__()
        pass
# SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)
    def build(self, input_shape):
        # print("GeneratorV2 input,",input_shape)
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        layers = tf.keras.layers

        self.layers.append( layers.Conv2D(32,(3,3),padding = 'same',kernel_initializer=init ))
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.PReLU() )
        self.layers.append( layers.Conv2D(64,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.PReLU() )
        self.layers.append( layers.Conv2D(64,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.PReLU() )

        for _ in range(4):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=64) )

        
        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )
        
        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  ) 
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )
        # self.layers.append( layers.Conv2D(64,(3,3),padding = 'same',kernel_initializer=init )  ) 
        # self.layers.append( layers.PReLU() )
        self.layers.append( layers.Conv2D(3,(1,1), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        # self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append(tf.keras.layers.Activation("tanh" , dtype='float32') )
        # self.layers.append( tf.keras.activations.tanh )

    def call(self,input_image):
        
        for layer in self.layers:
            input_image = layer(input_image)

        # print("GeneratorV2 output,",input_image.shape)
        return input_image







class UpsampleGenerator(tf.keras.layers.Layer):
    def __init__(self):
        super(UpsampleGenerator,self).__init__()
        pass
# SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)
    def build(self, input_shape):
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        layers = tf.keras.layers

        self.layers.append( layers.Conv2D(64,(3,3),padding = 'same',kernel_initializer=init ))
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.PReLU() )

        for _ in range(4):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=64) )


        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )
        
        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.PReLU() )

        self.layers.append( layers.Conv2D(3,(9,9), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        # self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append(tf.keras.layers.Activation("tanh" , dtype='float32') )

    def call(self,input_image):
        # print("UpsampleGenerator input, ",input_image.shape)
        for layer in self.layers:
            input_image = layer(input_image)
        # print("UpsampleGenerator output, ",input_image.shape)

        return input_image