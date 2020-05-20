import tensorflow as tf
from functools import partial
from config import Config

def decode_img(img,IMG_WIDTH, IMG_HEIGHT):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = img*2-1.0
    # resize the image to the desired size.
    img = tf.image.resize(img, [IMG_WIDTH+30, IMG_HEIGHT+30])
    img = tf.image.random_crop(img,[IMG_HEIGHT, IMG_WIDTH, 3] )

    return  tf.image.resize(img, [64,64]) ,  img

def process_path(file_path,IMG_WIDTH, IMG_HEIGHT):
#   label = get_label(file_path)
# load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img,IMG_WIDTH, IMG_HEIGHT)
    return img


def prepare_for_training(ds,BATCH_SIZE, cache=False, shuffle_buffer_size=300):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# usage: next(iter(data set))
def getAnimeCleanData(  BATCH_SIZE = 32 ,
                        IMG_WIDTH  = 512 ,
                        IMG_HEIGHT = 512 ):
    """[summary]

    Keyword Arguments:
        BATCH_SIZE {int} -- [description] (default: {32})
        IMG_WIDTH {int} -- [description] (default: {256})
        IMG_HEIGHT {int} -- [description] (default: {256})

    Returns:
        [PrefetchDataset] -- [description]
    """
    
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(Config.AnimeDataPath)

    # list_ds = tf.data.Dataset.list_files('../AnimeClean/*/*.jpg')
    process_path_width_height = partial(process_path,IMG_WIDTH = IMG_WIDTH,IMG_HEIGHT = IMG_HEIGHT)
    labeled_ds = list_ds.map(process_path_width_height, num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(labeled_ds,BATCH_SIZE)
    return train_ds



def decode_celeba_img(img,IMG_WIDTH, IMG_HEIGHT):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = img*2-1.0
    # resize the image to the desired size.
    img = tf.image.resize(img, [IMG_WIDTH+30, IMG_HEIGHT+30])
    img = tf.image.random_crop(img,[IMG_HEIGHT, IMG_WIDTH, 3] )
    return tf.image.resize(img, [64,64] )

def process_celeba(file_path,IMG_WIDTH, IMG_HEIGHT):
#   label = get_label(file_path)
# load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_celeba_img(img,IMG_WIDTH, IMG_HEIGHT)
    return img

def getCelebaData(  BATCH_SIZE = 32 ,
                        IMG_WIDTH  = 256 ,
                        IMG_HEIGHT = 256 ):
    """[summary]

    Keyword Arguments:
        BATCH_SIZE {int} -- [description] (default: {32})
        IMG_WIDTH {int} -- [description] (default: {256})
        IMG_HEIGHT {int} -- [description] (default: {256})

    Returns:
        [PrefetchDataset] -- [description]
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    list_ds = tf.data.Dataset.list_files(Config.CelebaDataPath)
    # process_path_width_height = partial(process_path,IMG_WIDTH = IMG_WIDTH,IMG_HEIGHT = IMG_HEIGHT)
    process_path_width_height = partial(process_celeba,IMG_WIDTH = IMG_WIDTH,IMG_HEIGHT = IMG_HEIGHT)

    labeled_ds = list_ds.map(process_path_width_height, num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(labeled_ds,BATCH_SIZE)
    return train_ds

