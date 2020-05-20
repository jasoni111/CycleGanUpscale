# experimental_run_v2<--------

import tensorflow as tf
import numpy as np


# import matplotlib.pyplot as plt
# import PIL
from data import getAnimeCleanData, getCelebaData
from loss import (
    w_d_loss,
    w_g_loss,
    cycle_loss,
    identity_loss,
    mse_loss,
    gradient_penalty,
)
from discriminator import W_Discriminator

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

    AnimeCleanData = getAnimeCleanData(BATCH_SIZE=1)
    CelebAData = getCelebaData(BATCH_SIZE=1)

    AnimeCleanData_iter = iter(AnimeCleanData)
    CelebAData_iter = iter(CelebAData)

    logdir = "./logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    generator_to_anime_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_to_human_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_anime_upscale_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_human_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_anime_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_anime_upscale_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    generator_to_anime = GeneratorV2()
    generator_to_human = GeneratorV2()

    generator_anime_upscale = UpsampleGenerator()

    # input: Batch, 256,256,3
    discriminator_human = W_Discriminator()
    discriminator_anime = W_Discriminator()

    discriminator_anime_upscale = W_Discriminator()

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(
        generator_to_anime=generator_to_anime,
        generator_to_human=generator_to_human,
        generator_anime_upscale=generator_anime_upscale,  # *
        discriminator_human=discriminator_human,
        discriminator_anime=discriminator_anime,
        discriminator_anime_upscale=discriminator_anime_upscale,  # *
        generator_to_anime_optimizer=generator_to_anime_optimizer,
        generator_to_human_optimizer=generator_to_human_optimizer,
        generator_anime_upscale_optimizer=generator_anime_upscale_optimizer,  # *
        discriminator_human_optimizer=discriminator_human_optimizer,
        discriminator_anime_optimizer=discriminator_anime_optimizer,
        discriminator_anime_upscale_optimizer=discriminator_anime_upscale_optimizer,  # *
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    @tf.function
    def init_pool(AnimeBatchImage, BigAnimeBatchImage, CelebaBatchImage):
        # store 30 images per type for training tool
        fake_anime_pool = generator_to_anime(CelebaBatchImage)
        cycled_anime_pool = generator_to_anime(generator_to_anime(AnimeBatchImage),)
        same_anime_pool = generator_to_anime(CelebaBatchImage)

        fake_human_pool = generator_to_human(AnimeBatchImage)
        cycled_human_pool = generator_to_human(generator_to_anime(CelebaBatchImage),)
        same_human_pool = generator_to_human(AnimeBatchImage)

        fake_anime_upscale_pool = generator_anime_upscale(fake_anime_pool)
        cycled_anime_upscale_pool = generator_anime_upscale(cycled_anime_pool)
        same_anime_upscale_pool = generator_anime_upscale(same_anime_pool)

        data_pools = [
            fake_anime_pool,
            cycled_anime_pool,
            same_anime_pool,
            fake_human_pool,
            cycled_human_pool,
            same_human_pool,
            fake_anime_upscale_pool,
            cycled_anime_upscale_pool,
            same_anime_upscale_pool,
        ]
        return data_pools

    AnimeBatchImage, BigAnimeBatchImage = next(AnimeCleanData_iter)
    CelebaBatchImage = next(CelebAData_iter)
    data_pools = init_pool(AnimeBatchImage, BigAnimeBatchImage, CelebaBatchImage)

    def add_data_to_pool(pool, new_data, pool_size=50):
        pool = new_data
        # tf.random.shuffle(pool)
        # pool = pool[:pool_size, :, :, :]
        # tf.concat([pool, new_data], 0)

    def get_data_from_pool(pool, batch_size=8):
        return pool[:batch_size, :, :, :]

    # out: Batch, 16, 16, 1
    # x is human, y is anime
    @tf.function
    def trainstep_G(real_human, real_anime, big_anime):
        with tf.GradientTape(persistent=True) as tape:
            fake_anime = generator_to_anime(real_human, training=True)
            cycled_human = generator_to_human(fake_anime, training=True)

            fake_human = generator_to_human(real_anime, training=True)
            cycled_anime = generator_to_anime(fake_human, training=True)

            same_human = generator_to_human(real_human, training=True)
            same_anime = generator_to_anime(real_anime, training=True)

            disc_fake_human = discriminator_human(fake_human, training=True)
            disc_fake_anime = discriminator_anime(fake_anime, training=True)

            fake_anime_upscale = generator_anime_upscale(fake_anime, training=True)
            real_anime_upscale = generator_anime_upscale(real_anime, training=True)

            cycled_anime_upscale= generator_anime_upscale(cycled_anime, training=True)
            same_anime_upscale = generator_anime_upscale(same_anime, training=True)

            disc_fake_upscale = discriminator_anime_upscale(
                fake_anime_upscale, training=True
            )
            disc_real_upscale = discriminator_anime_upscale(
                real_anime_upscale, training=True
            )
            # calculate the loss
            gen_anime_loss = w_g_loss(disc_fake_anime)
            gen_human_loss = w_g_loss(disc_fake_human)

            total_cycle_loss = cycle_loss(real_human, cycled_human) + cycle_loss(
                real_anime, cycled_anime
            )

            # Total generator loss = adversarial loss + cycle loss
            total_gen_anime_loss = (
                gen_anime_loss*1e3
                + total_cycle_loss
                + identity_loss(real_anime, same_anime)
            )

            tf.print("gen_anime_loss*1e3",gen_anime_loss*1e3)
            tf.print("total_cycle_loss",total_cycle_loss)
            tf.print("identity_loss",identity_loss(real_anime, same_anime))
            tf.print("--------------------------")
            total_gen_human_loss = (
                gen_human_loss*1e3
                + total_cycle_loss
                + identity_loss(real_human, same_human)
            )

            gen_upscale_loss = (
                w_g_loss(disc_fake_upscale)*1e3
                # + w_g_loss(disc_cycle_upscale)
                # + w_g_loss(disc_same_upscale)
                + identity_loss(big_anime, real_anime_upscale)* 1e-6
                # + identity_loss(big_anime, same_anime_upscale)
            )

            tf.print("w_g_loss(disc_fake_upscale)", w_g_loss(disc_fake_upscale))
            tf.print("identity_loss(big_anime, disc_real_upscale)", identity_loss(big_anime, disc_real_upscale))

            # tf.print("w_g_loss(disc_cycle_upscale)", w_g_loss(disc_cycle_upscale))
            # tf.print("w_g_loss(disc_same_upscale)", w_g_loss(disc_same_upscale))
            # tf.print(
            #     "identity_loss(big_anime, cycled_anime_upscale)",
            #     identity_loss(big_anime, cycled_anime_upscale),
            # )
            # tf.print(
            #     "identity_loss(big_anime, same_anime_upscale)",
            #     identity_loss(big_anime, same_anime_upscale),
            # )

            
        generator_to_anime_gradients = tape.gradient(
            total_gen_anime_loss, generator_to_anime.trainable_variables
        )
        generator_to_human_gradients = tape.gradient(
            total_gen_human_loss, generator_to_human.trainable_variables
        )
        generator_upscale_gradients = tape.gradient(
            gen_upscale_loss, generator_anime_upscale.trainable_variables
        )
        generator_to_anime_optimizer.apply_gradients(
            zip(generator_to_anime_gradients, generator_to_anime.trainable_variables)
        )
        generator_to_human_optimizer.apply_gradients(
            zip(generator_to_human_gradients, generator_to_human.trainable_variables)
        )
        generator_anime_upscale_optimizer.apply_gradients(
            zip(
                generator_upscale_gradients, generator_anime_upscale.trainable_variables
            )
        )

        return [
            real_human,
            real_anime,
            fake_anime,
            cycled_anime,
            same_anime,
            fake_human,
            cycled_human,
            same_human,
            fake_anime_upscale,
            cycled_anime_upscale,
            same_anime_upscale,
            gen_anime_loss,
            gen_human_loss,
            total_gen_anime_loss,
            total_gen_human_loss,
            gen_upscale_loss,
        ]

    @tf.function
    def trainstep_D(
        real_human,
        real_anime,
        big_anime,
        fake_anime,
        cycled_anime,
        same_anime,
        fake_human,
        cycled_human,
        same_human,
        fake_anime_upscale,
        cycled_anime_upscale,
        same_anime_upscale,
    ):
        with tf.GradientTape(persistent=True) as tape:
            disc_real_human = discriminator_human(real_human, training=True)
            disc_real_anime = discriminator_anime(real_anime, training=True)

            disc_fake_human = discriminator_human(fake_human, training=True)
            disc_fake_anime = discriminator_anime(fake_anime, training=True)

            disc_real_big = discriminator_anime_upscale(big_anime, training=True)
            disc_fake_upscale = discriminator_anime_upscale(
                fake_anime_upscale, training=True
            )
            # disc_same_upscale = discriminator_anime_upscale(
            #     same_anime_upscale, training=True
            # )

            discriminator_human_gradient_penalty = gradient_penalty(
                functools.partial(discriminator_human, training=True),
                real_human,
                fake_human,
            )
            discriminator_anime_gradient_penalty = gradient_penalty(
                functools.partial(discriminator_anime, training=True),
                real_anime,
                fake_anime,
            )
            discriminator_upscale_gradient_penalty = gradient_penalty(
                functools.partial(discriminator_human, training=True),
                big_anime,
                fake_anime_upscale,
            )

            disc_human_loss = (
                w_d_loss(disc_real_human, disc_fake_human)
                + discriminator_human_gradient_penalty
            )
            disc_anime_loss = (
                w_d_loss(disc_real_anime, disc_fake_anime)
                + discriminator_anime_gradient_penalty
            )
            disc_upscale_loss = (
                w_d_loss(disc_real_big, disc_fake_upscale)
                + discriminator_upscale_gradient_penalty
            )
            tf.print("disc_real_big",disc_real_big)
            tf.print("disc_fake_upscale",disc_fake_upscale)
            tf.print("disc_upscale_loss", disc_upscale_loss)
        

        discriminator_human_gradients = tape.gradient(
            disc_human_loss, discriminator_human.trainable_variables
        )
        discriminator_anime_gradients = tape.gradient(
            disc_anime_loss, discriminator_anime.trainable_variables
        )
        discriminator_upscale_gradients = tape.gradient(
            disc_upscale_loss, discriminator_anime_upscale.trainable_variables
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

    def process_data_for_display(input_image):
        return input_image * 0.5 + 0.5

    counter = 0
    i = -1

    print_string = [
        "real_human",
        "real_anime",
        "fake_anime",
        "cycled_anime",
        "same_anime",
        "fake_human",
        "cycled_human",
        "same_human",
        "fake_anime_upscale",
        "cycled_anime_upscale",
        "same_anime_upscale",
        "gen_anime_loss",
        "gen_human_loss",
        "total_gen_anime_loss",
        "total_gen_human_loss",
        "gen_upscale_loss",
    ]

    while True:
        i = i + 1
        counter = counter + 1
        AnimeBatchImage, BigAnimeBatchImage = next(AnimeCleanData_iter)
        CelebaBatchImage = next(CelebAData_iter)
        print(counter)

        # for j in range(3):
        result = trainstep_G(CelebaBatchImage, AnimeBatchImage, BigAnimeBatchImage)
        for j in range(9):
            add_data_to_pool(data_pools[j], result[2 + j])
        trainstep_D(
            CelebaBatchImage,
            AnimeBatchImage,
            BigAnimeBatchImage,
            *[get_data_from_pool(x) for x in data_pools]
        )
        # print("generator_to_anime.count_params()",generator_to_anime.count_params() )
        # print("generator_to_human.count_params()",generator_to_human.count_params() )
        # print("generator_anime_upscale.count_params()",generator_anime_upscale.count_params() )
        # print("discriminator_human.count_params()",discriminator_human.count_params() )
        # print("discriminator_anime.count_params()",discriminator_anime.count_params() )
        # print("discriminator_anime_upscale.count_params()",discriminator_anime_upscale.count_params() )

        if not (i % 5):
            with file_writer.as_default():
                for j in range(11):
                    tf.summary.image(
                        print_string[j], process_data_for_display(result[j]), step=i
                    )
                for j in range(11, len(print_string)):
                    tf.summary.scalar(print_string[j], result[j], step=i)
            ckpt_manager.save()


# testfun()
run_tensorflow()
