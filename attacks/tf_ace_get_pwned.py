import base64
import json

import h5py
import tensorflow as tf

new_model = tf.keras.models.load_model("tf.h5")

print("Transformers is not vulnerable to this, as it uses h5 directly.")
print("Keras uses a pickled code of the function within the `h5` attrs of the file")
print("Let's show you the marshalled code")

with h5py.File("tf_ace.h5") as f:
    data = json.loads(f.attrs["model_config"])
    print(base64.b64decode(data["config"]["layers"][-1]["config"]["function"][0]))
    pass
