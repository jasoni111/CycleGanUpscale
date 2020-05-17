import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
# import PIL
from data import getAnimeCleanData, getCelebaData
from loss import discriminator_loss, generator_loss, cycle_loss, identity_loss
from discriminator import Discriminator
from generator import Generator
from datetime import datetime


def run_tensorflow():
    """
    [summary] This is needed for tensorflow to free up my gpu ram...
    """
    

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)

    # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    # tf.keras.mixed_precision.experimental.set_policy(policy)
    # print('Compute dtype: %s' % policy.compute_dtype)
    # print('Variable dtype: %s' % policy.variable_dtype)


    AnimeCleanData = getAnimeCleanData(BATCH_SIZE=32)
    CelebaData = getCelebaData()

    logdir = "../logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    # AnimeBatchImage = next(iter(AnimeCleanData))
    # CelebaBatchImage = next(iter(CelebaData))
    # print(image.dtype)    

    # # checkpoint_path = "./checkpoints/train"

    # # ckpt = tf.train.Checkpoint(generator_to_anime=generator_to_anime,
    # #                            generator_to_real=generator_to_real,
    # #                            discriminator_x=discriminator_x,
    # #                            discriminator_y=discriminator_y,
    # #                            generator_to_anime_optimizer=generator_to_anime_optimizer,
    # #                            generator_to_real_optimizer=generator_to_real_optimizer,
    # #                            discriminator_x_optimizer=discriminator_x_optimizer,
    # #                            discriminator_y_optimizer=discriminator_y_optimizer)

    # # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # # # if a checkpoint exists, restore the latest checkpoint.
    # # if ckpt_manager.latest_checkpoint:
    # #   ckpt.restore(ckpt_manager.latest_checkpoint)
    # #   print ('Latest checkpoint restored!!')

    generator_to_anime_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_to_real_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # input: Batch, 256,256,3
    discriminator_x = Discriminator()
    discriminator_y = Discriminator()
    # out: Batch, 16, 16, 1

    generator_to_anime = Generator()
    generator_to_real = Generator()

# x is human, y is anime
    @tf.function
    def trainstep(real_human, real_anime):
        with tf.GradientTape(persistent = True) as tape:

            fake_anime = generator_to_anime(real_human, training=True)
            cycled_human = generator_to_real(fake_anime, training=True)

            fake_human = generator_to_real(real_anime, training=True)
            cycled_anime = generator_to_anime(fake_human, training=True)

            # same_human and same_anime are used for identity loss.
            same_human = generator_to_real(real_human, training=True)
            same_anime = generator_to_anime(real_anime, training=True)

            disc_real_human = discriminator_x(real_human, training=True)
            disc_real_anime = discriminator_y(real_anime, training=True)

            disc_fake_human = discriminator_x(fake_human, training=True)
            disc_fake_anime = discriminator_y(fake_anime, training=True)

            # calculate the loss
            gen_anime_loss = generator_loss(disc_fake_anime)
            gen_human_loss = generator_loss(disc_fake_human)
            
            total_cycle_loss = cycle_loss(real_human, cycled_human) + cycle_loss(real_anime, cycled_anime)
            
            # Total generator loss = adversarial loss + cycle loss
            total_gen_anime_loss = gen_anime_loss + total_cycle_loss + identity_loss(real_anime, same_anime)
            total_gen_human_loss = gen_human_loss + total_cycle_loss + identity_loss(real_human, same_human)

            disc_x_loss = discriminator_loss(disc_real_human, disc_fake_human)
            disc_y_loss = discriminator_loss(disc_real_anime, disc_fake_anime)

        # Calculate the gradients for generator and discriminator
        generator_to_anime_gradients = tape.gradient(total_gen_anime_loss, 
                                            generator_to_anime.trainable_variables)
        generator_to_human_gradients = tape.gradient(total_gen_human_loss, 
                                            generator_to_real.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                                discriminator_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                                discriminator_y.trainable_variables)

        # Apply the gradients to the optimizer
        generator_to_anime_optimizer.apply_gradients(zip(generator_to_anime_gradients, 
                                                generator_to_anime.trainable_variables))

        generator_to_real_optimizer.apply_gradients(zip(generator_to_human_gradients, 
                                                generator_to_real.trainable_variables))

        discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                    discriminator_x.trainable_variables))

        discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                    discriminator_y.trainable_variables))

        return fake_anime, cycled_human, fake_human, cycled_anime , same_human , same_anime, \
            gen_anime_loss, gen_human_loss, disc_x_loss, disc_y_loss, total_gen_anime_loss, total_gen_human_loss

    counter = 0
    while True:
        counter=counter+1
        AnimeBatchImage  = next(iter(AnimeCleanData))
        CelebaBatchImage = next(iter(CelebaData))

        

        

        if not(i % 5):
            fake_anime, cycled_human, fake_human, cycled_anime , same_human , same_anime, \
                gen_anime_loss, gen_human_loss, disc_x_loss, disc_y_loss, total_gen_anime_loss, total_gen_human_loss = trainstep(CelebaBatchImage, AnimeBatchImage)

            with file_writer.as_default():

                tf.summary.image("fake_anime", fake_anime, step=counter)
                tf.summary.image("cycled_human", cycled_human, step=counter)
                tf.summary.image("fake_human", fake_human, step=counter)
                tf.summary.image("cycled_anime", cycled_anime, step=counter)
                tf.summary.image("same_human", same_human, step=counter)
                tf.summary.image("same_anime", same_anime, step=counter)
                tf.summary.scalar("gen_anime_loss", gen_anime_loss, step=counter)
                tf.summary.scalar("gen_human_loss", gen_human_loss, step=counter)
                tf.summary.scalar("disc_x_loss", disc_x_loss, step=counter)
                tf.summary.scalar("disc_y_loss", disc_y_loss, step=counter)
                tf.summary.scalar("total_gen_anime_loss", total_gen_anime_loss, step=counter)
                tf.summary.scalar("total_gen_human_loss", total_gen_human_loss, step=counter)

                # tf.summary.image("CelebaBatchImage", CelebaBatchImage, step=counter)
        else:
            trainstep(CelebaBatchImage, AnimeBatchImage)


        

# testfun()
run_tensorflow()