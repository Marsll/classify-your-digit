import numpy as np
import tensorflow as tf

from mnist_model import cnn_model_fn


def predict(path_to_npy_file):
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir='./static/model',)

    picture = np.load(path_to_npy_file).reshape(28 * 28)
    picture = picture[np.newaxis, :]

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": picture},
        num_epochs=1,
        shuffle=False)

    out = mnist_classifier.predict(input_fn=predict_input_fn)
    out = [dict for dict in out]
    prediction = out[0]['classes']
    prediction_prob = out[0]['probabilities'][prediction]
    return prediction, np.around(prediction_prob * 100, decimals=1)
