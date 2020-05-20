import tensorflow as tf
# To import tensorflow_addons, I commented tensoflow_addons/activation/rrelu.py->Option[tf.random.Generator] away
# This only happens to tensoflow<=2.1
import tensorflow_addons as tfa


class StarDiscriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(StarDiscriminator,self).__init__()
        pass

    def build(self, input_shape):
        dim = input_shape[1]
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        self.layers = []
        
        mult = 2
        i = dim // 2

        self.layers.append(  tf.keras.layers.Conv2D(16, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        while i > 4 :
            self.layers.append(  tf.keras.layers.Conv2D(16 * mult, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
            self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
            i //=2
            mult *= 2

        self.logit_out_layer = tf.keras.layers.Conv2D(1, (4,4), padding='valid', kernel_initializer=init)
        self.label_out_layer = tf.keras.layers.Conv2D(1, (4,4), padding='valid', kernel_initializer=init) 
        self.out_layer = tf.keras.layers.Activation("linear" , dtype='float32')
        self.out_sigmoid_layer = tf.keras.layers.Activation("sigmoid" , dtype='float32')

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        logit = self.out_layer(self.logit_out_layer(inputs))
        label = self.out_sigmoid_layer(self.label_out_layer(inputs))

        return logit,label

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


class W_Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(W_Discriminator,self).__init__()
        pass

    def build(self, input_shape):
        dim = input_shape[1]
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        self.layers = []
        
        mult = 2
        i = dim // 2

        self.layers.append(  tf.keras.layers.Conv2D(16, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        while i > 4 :
            self.layers.append(  tf.keras.layers.Conv2D(16 * mult, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
            self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
            i //=2
            mult *= 2

        self.layers.append( tf.keras.layers.Conv2D(1, (4,4), padding='valid', kernel_initializer=init)) 
        self.layers.append(tf.keras.layers.Activation("linear" , dtype='float32') )

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
            tf.print("min_inputs",tf.keras.backend.min(inputs))
        tf.print("~~~~~~~~~~")
        return inputs