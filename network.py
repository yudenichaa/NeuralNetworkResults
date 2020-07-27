from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import numpy as np
from keras.models import load_model
from albumentations import Resize
from skimage.transform import resize
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


def get_model(model_path):
    session = tf.Session()
    graph = tf.get_default_graph()
    set_session(session)
    model = load_model(
        model_path,
        custom_objects={
            'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
            'iou_score': iou_score
        })
    return model, graph, session


def get_prediction(model, graph, session, image):
    with graph.as_default():
        set_session(session)
        image_data = image.astype(np.float32) / 255
        image_data = scale_and_reshape_image(image_data)
        prediction = model.predict(image_data)
        prediction_image = prediction.reshape(image_data.shape[1], image_data.shape[2])
        return prediction_image * 255


def scale_and_reshape_image(image_data, max_image_size=1500):
    width_scale_factor = image_data.shape[0] / max_image_size
    height_scale_factor = image_data.shape[1] / max_image_size
    scale_factor = max(width_scale_factor, height_scale_factor)
    if scale_factor > 1:
        image_data = resize(image_data,
                            (image_data.shape[0] // scale_factor,
                             image_data.shape[1] // scale_factor),
                            anti_aliasing=True)
    augmented = Resize(height=(image_data.shape[0] // 32) * 32,
                       width=(image_data.shape[1] // 32) * 32)(image=image_data)
    image_data = augmented['image']
    image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1], 3).astype(np.float32)
    return image_data

