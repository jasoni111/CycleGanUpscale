import tensorflow as tf

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# def w_discriminator_loss(real, generated):
#   real_loss = -tf.reduce_mean(real)
#   fake_loss = tf.reduce_mean(generated)
#   # real_loss = loss_obj(tf.ones_like(real), real)
#   # generated_loss = loss_obj(tf.zeros_like(generated), generated)
#   # total_disc_loss = real_loss + generated_loss
#   return real_loss + fake_loss

# def w_generator_loss(generated):
#   return -tf.reduce_mean(generated)
def w_d_loss(real_score,generated_score):
  real_loss = tf.reduce_mean(real_score)
  gen_loss = tf.reduce_mean(generated_score)
  return gen_loss - real_loss

def w_g_loss(generated_score):
  return -tf.reduce_mean(generated_score)



def _interpolate(a, b):
  shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
  alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
  inter = a + alpha * (b - a)
  inter.set_shape(a.shape)
  return inter

def gradient_penalty_star(f, real, fake):
  x = _interpolate(real, fake)
  with tf.GradientTape() as tape:
      tape.watch(x)
      pred,_ = f(x)
  grad = tape.gradient(pred, x)
  grad = grad
  norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
  # print("norm",tf.keras.backend.max(norm))
  gp = tf.reduce_mean((norm - 1.)**2)
  return gp

def gradient_penalty(f, real, fake):
  x = _interpolate(real, fake)
  with tf.GradientTape() as tape:
      tape.watch(x)
      pred = f(x)
  grad = tape.gradient(pred, x)
  grad = grad
  norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
  # print("norm",tf.keras.backend.max(norm))
  gp = tf.reduce_mean((norm - 1.)**2)
  return gp
# def gradient_penalty(f, real, fake):
#   alpha = tf.random.uniform([real.shape[0], 1, 1, 1], 0.0, 1.0)
#   inter_sample = fake * alpha + real * (1 - alpha)
#   with tf.GradientTape() as tape_gp:
#     tape_gp.watch(inter_sample)
#     inter_score = f(inter_sample)
#   gp_gradients = tape_gp.gradient(inter_score, inter_sample)
#   gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis = [1, 2, 3]))
#   tf.print("norm",tf.reduce_max(gp_gradients_norm))
#   gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)
#   return gp


  # return  _gradient_penalty(f, real, fake)

def discriminator_upscale_loss(real, generated,cycled,same):
  real_loss = loss_obj(tf.ones_like(real), real)
  zeros = tf.zeros_like(generated)
  generated_loss = loss_obj(zeros, generated)*3
  generated_loss += loss_obj(zeros, cycled)
  generated_loss += loss_obj(zeros, same)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5


def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1


def identity_loss(real_image, same_image):
  """[summary] 
  This try to maintain the same image if the domain are the same
  Arguments:
      real_image {[type]} -- [description]
      same_image {[type]} -- [description]

  Returns:
      [type] -- [description]
  """
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss


def mse_loss(real_image,fake_image):
  loss = tf.reduce_mean(tf.math.square(real_image - fake_image))
  return LAMBDA * 0.5 * loss