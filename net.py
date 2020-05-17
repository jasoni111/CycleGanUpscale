import tensorflow as tf
from config import Config
from utils import *


b_init = tf.constant_initializer(0.0) #bias initializer
w_init = tf.initializers.TruncatedNormal(stddev=0.02) #kernel_initializer

def batch_norm(axis=-1):
	return tf.keras.layers.BatchNormalization(momentum=Config.momentum,\
			epsilon=Config.epsilon,
			axis=axis)

def linear(units, use_bias=False):
	return tf.keras.layers.Dense(units = units,\
			kernel_initializer = w_init,
			use_bias=use_bias,
			bias_initializer=b_init)

def conv_layer(filters, kernel_size, strides, use_bias=False):

	return tf.keras.layers.Conv2D(filters = filters,\
			kernel_size = kernel_size,
			strides = strides,
			padding='same',
			data_format='channels_last',
			kernel_initializer = w_init,
			use_bias=use_bias,
			bias_initializer=b_init )

def transpose_conv_layer(filters, kernel_size, strides,use_bias=False):
	
	return tf.keras.layers.Conv2DTranspose(filters= filters,\
			kernel_size =kernel_size,
			strides = strides,
			padding='same',
			data_format='channels_last',
			kernel_initializer = w_init,
			use_bias=use_bias,
			bias_initializer=b_init)

class ConvBlock(tf.keras.Model):

	def __init__(self, filters):
		super(ConvBlock, self).__init__(name='conv_block')
		self.kernel_size = (5,5)
		self.strides = (2,2)
		self.filters = filters
		self.conv = conv_layer(self.filters, self.kernel_size, self.strides)
		self.batch_norm = batch_norm()
		self.leaky_ReLU = tf.keras.layers.LeakyReLU(0.2)

	def call(self, inputs, training=False):

		inputs = self.conv(inputs)
		inputs = self.batch_norm(inputs, training=training)
		output = self.leaky_ReLU(inputs)
		return output

class TransposeConvBlock(tf.keras.Model):

	def __init__(self, filters):
		super(TransposeConvBlock, self).__init__(name='transpose_conv_block')
		self.kernel_size = (5,5)
		self.strides = (2,2)
		self.filters = filters
		self.transpose_conv = transpose_conv_layer(self.filters, self.kernel_size, self.strides)
		self.batch_norm = batch_norm()
		self.ReLU = tf.keras.layers.ReLU()

	def call(self, inputs, training=False):

		inputs = self.transpose_conv(inputs)
		inputs = self.batch_norm(inputs, training=training)
		output = self.ReLU(inputs)
		return output


class Discriminator(tf.keras.Model):
	# class for discriminator
	def __init__(self):
		super(Discriminator, self).__init__(name='Discriminator')
		self.disc_filters = Config.disc_filters
		self.conv = conv_layer(filters=self.disc_filters[0], kernel_size=(5,5), strides=(2,2), use_bias=True)
		self.leaky_ReLU = tf.keras.layers.LeakyReLU(0.2)
		self.conv_blocks = [ConvBlock(self.disc_filters[i]) for i in range(1, len(self.disc_filters), 1)]
		self.flatten = tf.keras.layers.Flatten()
		self.linear = linear(units=1, use_bias=True)

	def call(self, inputs, training=False):
		inputs = self.conv(inputs)
		inputs = self.leaky_ReLU(inputs)
		for conv_block in self.conv_blocks:
			inputs = conv_block(inputs, training=training)
		inputs = self.flatten(inputs)
		output = self.linear(inputs)
		return output

class Generator(tf.keras.Model):
	# class for generator
	def __init__(self):
		super(Generator, self).__init__(name='Generator')
		self.gen_filters = Config.gen_filters
		self.img_shape = Config.img_shape
		self.linear = linear(units=4*4*self.gen_filters[0])
		self.batch_norm = batch_norm()
		self.ReLU = tf.keras.layers.ReLU()
		self.transpose_conv_blocks = [TransposeConvBlock(self.gen_filters[i]) for i in range(1, len(self.gen_filters), 1)]
		self.transpose_conv = transpose_conv_layer(filters=self.img_shape[-1], kernel_size=(5,5), strides=(2,2), use_bias=True)


	def call(self, inputs, training=False):
		inputs = self.linear(inputs)
		inputs = tf.reshape(inputs, [-1, 4, 4, self.gen_filters[0]])
		inputs = self.batch_norm(inputs, training=training)
		inputs = self.ReLU(inputs)
		for transpose_conv_block in self.transpose_conv_blocks:
			inputs = transpose_conv_block(inputs, training=training)
		inputs = self.transpose_conv(inputs)
		output = tf.nn.tanh(inputs)
		return output


class DCGAN:

	def __init__(self, restore):

		self.z_dim = Config.latent_dim
		self.global_batchsize = Config.global_batchsize

		self.gen_model = Generator()
		self.disc_model = Discriminator()
		self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=Config.gen_lr, beta_1=Config.beta1, beta_2=Config.beta2)
		self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=Config.disc_lr, beta_1=Config.beta1, beta_2=Config.beta2)
		self.train_writer = tf.summary.create_file_writer(Config.summaryDir+'train')

		self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),\
					generator_optimizer=self.gen_optimizer,
					generator_model = self.gen_model,
					discriminator_optimizer=self.disc_optimizer,
					discriminator_model=self.disc_model)

		self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, Config.modelDir, max_to_keep=3)

		self.global_step = 0

		if(restore):
			latest_ckpt= tf.train.latest_checkpoint(Config.modelDir)
			if not latest_ckpt:
				raise Exception('No saved model found in: ' + Config.modelDir)
			self.ckpt.restore(latest_ckpt)
			self.global_step = int(latest_ckpt.split('-')[-1])   # .../ckpt-300 returns 300 previously trained totalbatches
			print("Restored saved model from latest checkpoint")

	def compute_loss(self, labels, predictions):

		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,\
						reduction=tf.keras.losses.Reduction.NONE)
		return cross_entropy(labels, predictions)

	def disc_loss(self, real_output, fake_output):

		real_loss = self.compute_loss(tf.ones_like(real_output), real_output)
		fake_loss = self.compute_loss(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		total_loss = total_loss/self.global_batchsize
		return total_loss

	def gen_loss(self, fake_output):

		gen_loss = self.compute_loss(tf.ones_like(fake_output), fake_output)
		gen_loss = gen_loss / self.global_batchsize
		return gen_loss

	#@tf.function
	def train_step(self, real_imgs):

		noise = tf.random.normal(shape=[tf.shape(real_imgs)[0], self.z_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_imgs = self.gen_model(noise, training=True)
			real_output = self.disc_model(real_imgs, training=True)
			fake_output = self.disc_model(generated_imgs, training=True)
			d_loss = self.disc_loss(real_output, fake_output)
			g_loss = self.gen_loss(fake_output)

		G_grads = gen_tape.gradient(g_loss, self.gen_model.trainable_variables)
		D_grads = disc_tape.gradient(d_loss, self.disc_model.trainable_variables)

		self.gen_optimizer.apply_gradients(zip(G_grads, self.gen_model.trainable_variables))
		self.disc_optimizer.apply_gradients(zip(D_grads, self.disc_model.trainable_variables))

		#run g_optim twice to make sure d_loss doesn't go to zero
		with tf.GradientTape() as gen_tape:
			generated_imgs = self.gen_model(noise, training=True)
			fake_output = self.disc_model(generated_imgs, training=True)
			g_loss = self.gen_loss(fake_output)

		G_grads = gen_tape.gradient(g_loss, self.gen_model.trainable_variables)
		self.gen_optimizer.apply_gradients(zip(G_grads, self.gen_model.trainable_variables))

		return g_loss, d_loss

	#@tf.function
	def gen_step(self, random_latents):
		gen_imgs = self.gen_model(random_latents, training=False)
		return gen_imgs


	@tf.function
	def train_loop(self, num_epochs, dist_dataset, dist_noise_dataset):

		num_batches = self.global_step
		for i in range(num_epochs):
			print('At Epoch {}'.format(i+1))
			print('.........................................')
			for one_batch in dist_dataset:
				total_g_loss, total_d_loss = self.train_step(one_batch)

				with self.train_writer.as_default():
					tf.summary.scalar('generator_loss',total_g_loss, step=num_batches)
					tf.summary.scalar('discriminator_loss',total_d_loss, step=num_batches)

				if(num_batches % Config.image_snapshot_freq == 0):
					for dist_gen_noise in dist_noise_dataset:
						gen_imgs = self.gen_step(dist_gen_noise)

					filename = Config.results_dir + 'fakes_epoch{:02d}_batch{:05d}.jpg'.format(i+1, num_batches)
					# save_image_grid(gen_imgs.numpy(), filename, drange=[-1,1], grid_size=Config.grid_size)

				num_batches+=1
				print('Gen_loss at batch {}: {:0.3f}'.format(num_batches, total_g_loss))
				print('Disc_loss at batch {}: {:0.3f}'.format(num_batches, total_d_loss))

			self.ckpt.step.assign(i+1)
			self.ckpt_manager.save()