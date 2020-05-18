import tensorflow as tf
# To import tensorflow_addons, I commented tensoflow_addons/activation/rrelu.py->Option[tf.random.Generator] away
# This only happens to tensoflow<=2.1
import tensorflow_addons as tfa

class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        self.layers.append(  tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        self.layers.append(  tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init) )
        self.layers.append(  tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        self.layers.append(  tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init) )
        self.layers.append(  tfa.layers.InstanceNormalization(axis=-1) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        # self.layers.append(  tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        # self.layers.append(  tfa.layers.InstanceNormalization(axis=-1))
        # self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2))

        self.layers.append(  tf.keras.layers.Conv2D(256, (4,4), padding='same', kernel_initializer=init))
        self.layers.append(  tfa.layers.InstanceNormalization(axis=-1))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2))
        # patch output
        self.layers.append( tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)) 
        self.layers.append(tf.keras.layers.Activation("linear" , dtype='float32') )
        # self.layers.append(tf.keras.activations.linear(dtype='float32') )

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

# class UpScaleDiscriminator(tf.keras.layers.Layer):
#     def __init__(self):
#         super(UpScaleDiscriminator,self).__init__()
#         pass

#     def build(self, input_shape):
#         self.layers = []

#         init = tf.keras.initializers.RandomNormal(stddev=0.02)
        
#         self.layers.append(  tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
#         self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

#         self.layers.append(  tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init) )
#         self.layers.append(  tfa.layers.InstanceNormalization(axis=-1) )
#         self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

#         self.layers.append(  tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init) )
#         self.layers.append(  tfa.layers.InstanceNormalization(axis=-1) )
#         self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

#         self.layers.append(  tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
#         self.layers.append(  tfa.layers.InstanceNormalization(axis=-1))
#         self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2))

#         self.layers.append(  tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init))
#         self.layers.append(  tfa.layers.InstanceNormalization(axis=-1))
#         self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2))
#         # patch output
#         self.layers.append( tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init)) 

#     def call(self, inputs):
#         for layer in self.layers:
#             inputs = layer(inputs)
#         return inputs