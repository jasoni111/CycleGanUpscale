import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt
# import PIL
from data import getAnimeCleanData, getCelebaData
from loss import (
    discriminator_loss,
    generator_loss,
    cycle_loss,
    identity_loss,
    mse_loss,
    gradient_penalty_star,
)
from discriminator import StarDiscriminator, Discriminator

# , UpScaleDiscriminator
from generator import GeneratorV2, UpsampleGenerator
from datetime import datetime
batch_size = 32

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

    mixed_precision = tf.keras.mixed_precision.experimental

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)

    AnimeCleanData = getAnimeCleanData(BATCH_SIZE=batch_size)
    CelebaData = getCelebaData(BATCH_SIZE=batch_size)

    logdir = "./logs/Startrain_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    generator_optimizer = mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5), loss_scale="dynamic"
    )

    discriminator_optimizer = mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5), loss_scale="dynamic"
    )

    up_G_optim = mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5), loss_scale="dynamic"
    )
    up_D_optim = mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5), loss_scale="dynamic"
    )
    up_G = UpsampleGenerator()
    up_D = Discriminator()

    generator = GeneratorV2()
    # input: Batch, 256,256,3
    discriminator = StarDiscriminator()

    checkpoint_path = "./checkpoints/StarTrain"

    ckpt = tf.train.Checkpoint(
        generator = generator,
        discriminator = discriminator,
        generator_optimizer = generator_optimizer,
        discriminator_optimizer = discriminator_optimizer,

    )

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
            ones = tf.ones_like(real_human)
            neg_ones = tf.ones_like(real_human) * -1

            def get_domain_anime(img):
                return tf.concat([img, ones], 3)

            def get_domain_human(img):
                return tf.concat([img, neg_ones], 3)

            fake_anime = generator(get_domain_anime(real_human), training=True)
            cycled_human = generator(get_domain_human(fake_anime), training=True)

            fake_human = generator(get_domain_human(real_anime), training=True)
            cycled_anime = generator(get_domain_anime(fake_human), training=True)

            # same_human and same_anime are used for identity loss.
            same_human = generator(get_domain_human(real_human), training=True)
            same_anime = generator(get_domain_anime(real_anime), training=True)

            disc_real_human, label_real_human = discriminator(real_human, training=True)
            disc_real_anime, label_real_anime = discriminator(real_anime, training=True)

            disc_fake_human, label_fake_human = discriminator(fake_human, training=True)
            disc_fake_anime, label_fake_anime = discriminator(fake_anime, training=True)

            _, label_cycled_human = discriminator(cycled_human, training=True)
            _, label_cycled_anime = discriminator(cycled_anime, training=True)

            _, label_same_human = discriminator(same_human, training=True)
            _, label_same_anime = discriminator(same_anime, training=True)

            # calculate the loss
            gen_anime_loss = generator_loss(disc_fake_anime)
            gen_human_loss = generator_loss(disc_fake_human)

            total_cycle_loss = cycle_loss(real_human, cycled_human) + cycle_loss(
                real_anime, cycled_anime
            )

            gen_class_loss = (
                discriminator_loss(label_fake_human, label_fake_anime)
                + discriminator_loss(label_cycled_human, label_cycled_anime)
                + discriminator_loss(label_same_human, label_same_anime)
            )

            # Total generator loss = adversarial loss + cycle loss
            total_gen_loss = (
                gen_anime_loss
                + gen_human_loss 
                + gen_class_loss
                + total_cycle_loss * 0.1
                + identity_loss(real_anime, same_anime)
                + identity_loss(real_human, same_human)
            )

            tf.print("gen_anime_loss",gen_anime_loss)
            tf.print("gen_human_loss",gen_human_loss)
            tf.print("gen_class_loss",gen_class_loss)
            tf.print("total_cycle_loss",total_cycle_loss)
            tf.print("identity_loss(real_anime, same_anime)",identity_loss(real_anime, same_anime))
            tf.print("identity_loss(real_human, same_human)",identity_loss(real_human, same_human))

            scaled_total_gen_anime_loss = generator_optimizer.get_scaled_loss(
                total_gen_loss
            )

            disc_human_loss = discriminator_loss(disc_real_human, disc_fake_human)
            disc_anime_loss = discriminator_loss(disc_real_anime, disc_fake_anime)

            # disc_gp_anime = gradient_penalty_star(partial(discriminator, training=True), real_anime,fake_anime )
            # disc_gp_human = gradient_penalty_star(partial(discriminator, training=True), real_human,fake_human )

            disc_loss = disc_human_loss + disc_anime_loss + discriminator_loss(label_real_human,label_real_anime)
            # +disc_gp_anime+disc_gp_human

            scaled_disc_loss = discriminator_optimizer.get_scaled_loss(
                disc_loss
            )

        # Calculate the gradients for generator and discriminator
        generator_gradients =generator_optimizer.get_unscaled_gradients( tape.gradient(
            scaled_total_gen_anime_loss, generator.trainable_variables
        ))
        discriminator_gradients = discriminator_optimizer.get_unscaled_gradients( tape.gradient(
            scaled_disc_loss, discriminator.trainable_variables
        ))

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )

        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )

        with tf.GradientTape(persistent=True) as tape:
            real_anime_up = up_G(real_anime)
            fake_anime_up = up_G(fake_anime)

            dis_fake_anime_up = up_D(fake_anime_up)
            dis_real_anime_up = up_D(real_anime_up)
            dis_ori_anime = up_D(big_anime)
            gen_up_loss =  generator_loss(fake_anime_up) + generator_loss(dis_real_anime_up)*0.1
            dis_up_loss = discriminator_loss(dis_ori_anime,dis_fake_anime_up)+discriminator_loss(dis_ori_anime,dis_real_anime_up)*0.1
            scaled_gen_up_loss = up_G_optim.get_scaled_loss(gen_up_loss)
            scaled_disc_loss = up_D_optim.get_scaled_loss(dis_up_loss)

        up_G_gradients =up_G_optim.get_unscaled_gradients( tape.gradient(
            scaled_gen_up_loss, up_G.trainable_variables
        ))
        up_D_gradients = up_D_optim.get_unscaled_gradients( tape.gradient(
            scaled_disc_loss, up_D.trainable_variables
        ))

        up_G_optim.apply_gradients(
            zip(up_G_gradients, up_G.trainable_variables)
        )

        up_D_optim.apply_gradients(
            zip(up_D_gradients, up_D.trainable_variables)
        )
            

        return (
            real_human,
            real_anime,
            fake_anime,
            cycled_human,
            fake_human,
            cycled_anime,
            same_human,
            same_anime,
            fake_anime_up,
            real_anime_up,
            gen_anime_loss,
            gen_human_loss,
            disc_human_loss,
            disc_anime_loss,
            gen_up_loss,
            dis_up_loss
        )

    def process_data_for_display(input_image):
        return input_image * 0.5 + 0.5


    print_string = [
            "real_human",
            "real_anime",
            "fake_anime",
            "cycled_human",
            "fake_human",
            "cycled_anime",
            "same_human",
            "same_anime",
            "fake_anime_up",
            "real_anime_up",
            "gen_anime_loss",
            "gen_human_loss",
            "disc_human_loss",
            "disc_anime_loss",
            "gen_up_loss",
            "dis_up_loss"
    ]

    counter = 0
    i = -1
    while True:
        i = i + 1
        counter = counter + 1
        AnimeBatchImage, BigAnimeBatchImage = next(iter(AnimeCleanData))
        CelebaBatchImage = next(iter(CelebaData))
        print(counter)

        if not (i % 5):
            result = trainstep(CelebaBatchImage, AnimeBatchImage,BigAnimeBatchImage)

            with file_writer.as_default():
                for j in range(len(result)):
                    if j<10:
                        tf.summary.image(
                        print_string[j],
                        process_data_for_display(result[j]),
                        step=counter,
                        )
                    else:
                        tf.summary.scalar(
                        print_string[j],
                        result[j],
                        step=counter,
                        )
                
            ckpt_manager.save()
        else:
            trainstep(CelebaBatchImage, AnimeBatchImage,BigAnimeBatchImage)


# testfun()
run_tensorflow()
