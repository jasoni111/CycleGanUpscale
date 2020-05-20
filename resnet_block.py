import tensorflow as tf
import tensorflow_addons as tfa


class ResNetBlockInstanceNorm(tf.keras.layers.Layer):
    def __init__(self,num_filter):
        super(ResNetBlockInstanceNorm,self).__init__()
        self.num_filter = num_filter
        pass
    def build(self,image_shape):
        print("resblock constructing...:",image_shape)
        # self.num_filter = image_shape[3]
        init =  tf.keras.initializers.RandomNormal(stddev=0.02)
        self.cov1 = tf.keras.layers.Conv2D(self.num_filter, (3,3), padding = 'same',kernel_initializer=init)
        self.i_norm1 =  tfa.layers.InstanceNormalization(axis=-1)
        self.relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.cov2 = tf.keras.layers.Conv2D(self.num_filter, (3,3), padding = 'same',kernel_initializer=init)
        self.i_norm2 =  tfa.layers.InstanceNormalization(axis=-1)
        self.sum = tf.keras.layers.Add()

    def call(self, image):
        y = self.cov1(image)
        y = self.i_norm1(y)
        y = self.relu(y)
        y = self.cov2(y)
        y = self.i_norm2(y)
        return self.sum([y,image])

# (tf.keras.layers.Layer):
#     def __init__(self):
#         super(Discriminator,self).__init__()
#         pass

# def resnet_block_instance_Norm(n_filters, input_layer):
# 	# weight initialization
# 	init =  tf.keras.initializers.RandomNormal(stddev=0.02)
# 	# first layer convolutional layer
# 	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
# 	g = InstanceNormalization(axis=-1)(g)
# 	g = Activation('relu')(g)
# 	# second convolutional layer
# 	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
# 	g = InstanceNormalization(axis=-1)(g)
# 	# concatenate merge channel-wise with input layer
# 	g = Concatenate()([g, input_layer])
# 	return g