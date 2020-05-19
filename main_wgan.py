# experimental_run_v2<--------

import tensorflow as tf
import numpy as np



# import matplotlib.pyplot as plt
# import PIL
from data import getAnimeCleanData, getCelebaData
from loss import w_d_loss, w_g_loss, cycle_loss, identity_loss, mse_loss, gradient_penalty
from discriminator import Discriminator

# , UpScaleDiscriminator
from generator import GeneratorV2, UpsampleGenerator
from datetime import datetime
import functools


def run_tensorflow():
    """
    [summary] This is needed for tensorflow to free up my gpu ram...
    """

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # mixed_precision = tf.keras.mixed_precision.experimental

    # policy = mixed_precision.Policy("mixed_float16")
    # mixed_precision.set_policy(policy)

    AnimeCleanData = getAnimeCleanData(BATCH_SIZE=16)
    CelebaData = getCelebaData(BATCH_SIZE=16)

    logdir = "./logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    # AnimeBatchImage = next(iter(AnimeCleanData))
    # CelebaBatchImage = next(iter(CelebaData))
    # print(image.dtype)

    # # # if a checkpoint exists, restore the latest checkpoint.
    # # if ckpt_manager.latest_checkpoint:
    # #   ckpt.restore(ckpt_manager.latest_checkpoint)
    # #   print ('Latest checkpoint restored!!')

    generator_to_anime_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_to_real_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    generator_anime_upscale_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_human_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_anime_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    discriminator_anime_upscale_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    generator_to_anime = GeneratorV2()
    generator_to_real = GeneratorV2()

    generator_anime_upscale = UpsampleGenerator()

    # input: Batch, 256,256,3
    discriminator_human = Discriminator()
    discriminator_anime = Discriminator()

    discriminator_anime_upscale = Discriminator()

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(
        generator_to_anime=generator_to_anime,
        generator_to_real=generator_to_real,
        generator_anime_upscale=generator_anime_upscale,  # *
        discriminator_human=discriminator_human,
        discriminator_anime=discriminator_anime,
        discriminator_anime_upscale=discriminator_anime_upscale,  # *
        generator_to_anime_optimizer=generator_to_anime_optimizer,
        generator_to_real_optimizer=generator_to_real_optimizer,
        generator_anime_upscale_optimizer=generator_anime_upscale_optimizer,  # *
        discriminator_human_optimizer=discriminator_human_optimizer,
        discriminator_anime_optimizer=discriminator_anime_optimizer,
        discriminator_anime_upscale_optimizer=discriminator_anime_upscale_optimizer,  # *
    )

    # print("generator_to_anime",generator_to_anime.trainable_variables)
    # print("generator_to_real",generator_to_real.trainable_variables)
    # print("generator_anime_upscale",generator_anime_upscale.trainable_variables)
    # print("discriminator_human",discriminator_human.trainable_variables)
    # print("discriminator_anime",discriminator_anime.trainable_variables)
    # print("discriminator_anime_upscale",discriminator_anime_upscale.trainable_variables)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    # out: Batch, 16, 16, 1
    # x is human, y is anime
    @tf.function
    def trainstep(real_human, real_anime, big_anime):
        with tf.GradientTape(persistent=True) as tape:

            fake_anime = generator_to_anime(real_human, training=True)
            cycled_human = generator_to_real(fake_anime, training=True)

            fake_human = generator_to_real(real_anime, training=True)
            cycled_anime = generator_to_anime(fake_human, training=True)

            # same_human and same_anime are used for identity loss.
            same_human = generator_to_real(real_human, training=True)
            same_anime = generator_to_anime(real_anime, training=True)

            disc_real_human = discriminator_human(real_human, training=True)
            disc_real_anime = discriminator_anime(real_anime, training=True)

            disc_fake_human = discriminator_human(fake_human, training=True)
            disc_fake_anime = discriminator_anime(fake_anime, training=True)

            fake_anime_upscale = generator_anime_upscale(fake_anime)
            real_anime_upscale = generator_anime_upscale(real_anime)

            disc_fake_upscale = discriminator_anime_upscale(fake_anime_upscale)
            disc_real_upscale = discriminator_anime_upscale(real_anime_upscale)
            disc_real_big = discriminator_anime_upscale(big_anime)

            # calculate the loss
            gen_anime_loss = w_g_loss(disc_fake_anime)
            gen_human_loss = w_g_loss(disc_fake_human)

            total_cycle_loss = cycle_loss(real_human, cycled_human) + cycle_loss(
                real_anime, cycled_anime
            )

            # Total generator loss = adversarial loss + cycle loss
            total_gen_anime_loss = (
                gen_anime_loss
                + total_cycle_loss
                + identity_loss(real_anime, same_anime)
            )

            total_gen_human_loss = (
                gen_human_loss
                + total_cycle_loss
                + identity_loss(real_human, same_human)
            )
            
            gen_upscale_loss = (
                  w_g_loss(disc_fake_upscale)
                +  w_g_loss(disc_real_upscale)
                + mse_loss(big_anime, real_anime_upscale)
                + identity_loss(big_anime, real_anime_upscale)
            )
            

            discriminator_human_gradient_penalty = gradient_penalty(functools.partial(discriminator_human, training=True), real_human, fake_human)*10
            discriminator_anime_gradient_penalty = gradient_penalty(functools.partial(discriminator_anime, training=True), real_anime, fake_anime)*10
            discriminator_upscale_gradient_penalty = gradient_penalty(functools.partial(discriminator_human, training=True), big_anime,fake_anime_upscale )*5
            discriminator_upscale_gradient_penalty += gradient_penalty(functools.partial(discriminator_human, training=True), big_anime,real_anime_upscale )*5

            disc_human_loss =  w_d_loss(disc_real_human, disc_fake_human) + discriminator_human_gradient_penalty
            disc_anime_loss =  w_d_loss(disc_real_anime, disc_fake_anime) + discriminator_anime_gradient_penalty
            # # print("ggg",big_anime.shape)
            disc_upscale_loss =  w_d_loss(disc_real_big, disc_fake_upscale) 
            disc_upscale_loss += w_d_loss(disc_real_big, disc_real_upscale) + discriminator_upscale_gradient_penalty

        # Calculate the gradients for generator and discriminator
        generator_to_anime_gradients = tape.gradient(
            total_gen_anime_loss, generator_to_anime.trainable_variables
        )
        generator_to_human_gradients = tape.gradient(
            total_gen_human_loss, generator_to_real.trainable_variables
        )

        discriminator_human_gradients = tape.gradient(
            disc_human_loss, discriminator_human.trainable_variables
        )
        discriminator_anime_gradients = tape.gradient(
            disc_anime_loss, discriminator_anime.trainable_variables
        )

        generator_upscale_gradients = tape.gradient(
            gen_upscale_loss, generator_anime_upscale.trainable_variables
        )

        discriminator_upscale_gradients = tape.gradient(
            disc_upscale_loss, discriminator_anime_upscale.trainable_variables
        )

        # Apply the gradients to the optimizer

        generator_to_anime_optimizer.apply_gradients(
            zip(generator_to_anime_gradients, generator_to_anime.trainable_variables)
        )

        generator_to_real_optimizer.apply_gradients(
            zip(generator_to_human_gradients, generator_to_real.trainable_variables)
        )
        
        generator_anime_upscale_optimizer.apply_gradients(
            zip(
                generator_upscale_gradients, generator_anime_upscale.trainable_variables
            )
        )

        discriminator_human_optimizer.apply_gradients(
            zip(discriminator_human_gradients, discriminator_human.trainable_variables)
        )


        discriminator_anime_optimizer.apply_gradients(
            zip(discriminator_anime_gradients, discriminator_anime.trainable_variables)
        )

        discriminator_anime_upscale_optimizer.apply_gradients(
            zip(
                discriminator_upscale_gradients,
                discriminator_anime_upscale.trainable_variables,
            )
        )

        return (
            fake_anime,
            cycled_human,
            fake_human,
            cycled_anime,
            same_human,
            same_anime,
            fake_anime_upscale,
            real_anime_upscale,
            gen_anime_loss,
            gen_human_loss,
            disc_human_loss,
            disc_anime_loss,
            total_gen_anime_loss,
            total_gen_human_loss,
            gen_upscale_loss,
            disc_upscale_loss,
        )

    def process_data_for_display(input_image):
        return input_image * 0.5 + 0.5

    counter = 0
    i = -1
    while True:
        i = i + 1
        counter = counter + 1
        AnimeBatchImage, BigAnimeBatchImage = next(iter(AnimeCleanData))
        CelebaBatchImage = next(iter(CelebaData))
        print(counter)

        if not (i % 5):

            (
                fake_anime,
                cycled_human,
                fake_human,
                cycled_anime,
                same_human,
                same_anime,
                fake_anime_upscale,
                real_anime_upscale,
                gen_anime_loss,
                gen_human_loss,
                disc_human_loss,
                disc_anime_loss,
                total_gen_anime_loss,
                total_gen_human_loss,
                gen_upscale_loss,
                disc_upscale_loss,
            ) = trainstep(CelebaBatchImage, AnimeBatchImage, BigAnimeBatchImage)

            with file_writer.as_default():
                tf.summary.image(
                    "AnimeBatchImage",
                    process_data_for_display(AnimeBatchImage),
                    step=counter,
                )
                tf.summary.image(
                    "CelebaBatchImage",
                    process_data_for_display(CelebaBatchImage),
                    step=counter,
                )
                tf.summary.image(
                    "fake_anime", process_data_for_display(fake_anime), step=counter
                )
                tf.summary.image(
                    "cycled_human", process_data_for_display(cycled_human), step=counter
                )
                tf.summary.image(
                    "fake_human", process_data_for_display(fake_human), step=counter
                )
                tf.summary.image(
                    "cycled_anime", process_data_for_display(cycled_anime), step=counter
                )
                tf.summary.image(
                    "same_human", process_data_for_display(same_human), step=counter
                )
                tf.summary.image(
                    "same_anime", process_data_for_display(same_anime), step=counter
                )

                tf.summary.image("fake_anime_upscale", fake_anime_upscale, step=counter)
                tf.summary.image("real_anime_upscale", real_anime_upscale, step=counter)

                tf.summary.scalar("gen_anime_loss", gen_anime_loss, step=counter)
                tf.summary.scalar("gen_human_loss", gen_human_loss, step=counter)
                tf.summary.scalar("disc_human_loss", disc_human_loss, step=counter)
                tf.summary.scalar("disc_anime_loss", disc_anime_loss, step=counter)
                tf.summary.scalar(
                    "total_gen_anime_loss", total_gen_anime_loss, step=counter
                )
                tf.summary.scalar(
                    "total_gen_human_loss", total_gen_human_loss, step=counter
                )
                tf.summary.scalar("gen_upscale_loss", gen_upscale_loss, step=counter)
                tf.summary.scalar("disc_upscale_loss", disc_upscale_loss, step=counter)

                # tf.summary.image("CelebaBatchImage", CelebaBatchImage, step=counter)

            ckpt_manager.save()
        else:
            trainstep(CelebaBatchImage, AnimeBatchImage, BigAnimeBatchImage)


# testfun()
run_tensorflow()
