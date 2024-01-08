import tensorflow as tf


def gpu_setup():
    # Limit tensorflow using all of the vram from gpu
    # Avoid OOM (Out Of Memory) erors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    print(len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)