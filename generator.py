import tensorflow as tf
from resnet_block import ResNetBlockInstanceNorm
import tensorflow_addons as tfa


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
        self.layers.append( layers.Conv2D(64,(7,7),padding = 'same',kernel_initializer=init ))
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2D(128,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2D(256,(7,7), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        for _ in range(3):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=256) )
        self.layers.append( layers.Conv2DTranspose(128,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2DTranspose(64,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( layers.ReLU() )
        self.layers.append( layers.Conv2D(3,(7,7), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        self.layers.append( tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append( tf.keras.activations.tanh )

    def call(self,input_image):
        print(input_image.shape)
        for layer in self.layers:
            input_image = layer(input_image)
            print(input_image.shape)

        return input_image