import tensorflow as tf
from suppixel import SubpixelConv2D


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
        self.i_norm1 =  tf.keras.layers.LayerNormalization()
        self.relu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.cov2 = tf.keras.layers.Conv2D(self.num_filter, (3,3), padding = 'same',kernel_initializer=init)
        self.i_norm2 =  tf.keras.layers.LayerNormalization()
        self.sum = tf.keras.layers.Add()

    def call(self, image):
        y = self.cov1(image)
        y = self.i_norm1(y)
        y = self.relu(y)
        y = self.cov2(y)
        y = self.i_norm2(y)
        return self.sum([y,image])


class GeneratorV2(tf.keras.layers.Layer):
    def __init__(self):
        super(GeneratorV2,self).__init__()
        pass
    def build(self, input_shape):
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        layers = tf.keras.layers

        self.layers.append( layers.Conv2D(32,(3,3),padding = 'same',kernel_initializer=init ))
        self.layers.append( layers.LayerNormalization() )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )
        self.layers.append( layers.Conv2D(64,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( layers.LayerNormalization() )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )
        self.layers.append( layers.Conv2D(64,(3,3), strides=(2,2),padding = 'same',kernel_initializer=init )  )
        self.layers.append( layers.LayerNormalization() )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )

        for _ in range(4):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=64) )

        
        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )
        
        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  ) 
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.LeakyReLU(alpha=0.2) )

        self.layers.append( layers.Conv2D(3,(3,3), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        self.layers.append(tf.keras.layers.Activation("tanh" , dtype='float32') )

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
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.ReLU() )

        for _ in range(2):
            self.layers.append(ResNetBlockInstanceNorm(num_filter=64) )

        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.ReLU() )
        
        self.layers.append( layers.Conv2D(256,(3,3),padding = 'same',kernel_initializer=init )  )
        self.layers.append( SubpixelConv2D(input_shape,scale=2) )
        self.layers.append( layers.ReLU() )

        self.layers.append( layers.Conv2D(3,(3,3), strides=(1,1),padding = 'same',kernel_initializer=init )  )
        self.layers.append(tf.keras.layers.Activation("tanh" , dtype='float32') )

    def call(self,input_image):
        # print("UpsampleGenerator input, ",input_image.shape)
        for layer in self.layers:
            input_image = layer(input_image)
            # print(layer)
            # print(layer.count_params())
        # print("UpsampleGenerator output, ",input_image.shape)

        return input_image