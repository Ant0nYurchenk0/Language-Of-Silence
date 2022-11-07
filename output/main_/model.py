import os
import cv2 
import pathlib
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder

PROGRAM_PATH = "\\".join(pathlib.Path(__file__).parent.resolve().parts)+"\\_RealTimeObjectDetection"
WORKSPACE_PATH = PROGRAM_PATH+"\\Tensorflow"+ "\\workspace"
SCRIPTS_PATH = PROGRAM_PATH+"\\Tensorflow"+ "\\scripts"
APIMODEL_PATH = PROGRAM_PATH+"\\Tensorflow"+ "\\models"
ANNOTATION_PATH = WORKSPACE_PATH+"\\annotations"
IMAGE_PATH = WORKSPACE_PATH + "\\images"
MODEL_PATH = WORKSPACE_PATH+ "\\models"
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+ "\\pre-trained-models"
CONFIG_PATH = MODEL_PATH+ "\\my_ssd_mobnet"+ "\\pipeline.config"
CHECKPOINT_PATH = MODEL_PATH+ "\\my_ssd_mobnet"
CUSTOM_MODEL_NAME = "my_ssd_mobnet" 
CONFIG_PATH = MODEL_PATH+ "\\"+CUSTOM_MODEL_NAME+ "\\pipeline.config"

configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs["model"], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, "ckpt-11")).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+"\label_map.pbtxt")


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
    
capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = capture.read()
image_np = np.array(frame)
  
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)