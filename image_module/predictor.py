# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import base64
import torch
import cv2

import flask
import numpy as np

from flask_cors import CORS
from mask import create_mask
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps


prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ImageModule(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """
        Get the model object for this instance, loading it if it's not already loaded.
        """
        if cls.model == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cls.model = torch.jit.load(os.path.join(model_path, "image-inpainter.pt"), map_location=device)
            cls.model.eval()
        return cls.model

    @classmethod
    def predict(cls, image, mask):
        """
        For the input, do the predictions and return them.
        """
        clf = cls.get_model()
        return clf(image, mask)

    @classmethod
    def string_to_image(cls, base64_string):
        """
        Convert bytes streams to Image.
        """
        imgdata = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(imgdata)).convert('RGB')

    @classmethod
    def resize_with_padding(cls, image, target_size):
        """
        Return the resized image. (target_size x target_size)
        """
        if max(image.size) > target_size:
            ratio = max(image.size) / target_size
            width = round(image.size[0] / ratio)
            height = round(image.size[1] / ratio)
            image = image.resize((width, height), Image.ANTIALIAS)
            assert max(image.size) == target_size

        delta_width = target_size - image.size[0]
        delta_height = target_size - image.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        
        return ImageOps.expand(image, padding, fill=(255, 255, 255))

# The flask app for serving predictions
app = flask.Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

@app.route("/ping", methods=["GET"])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.
    """
    health = ImageModule.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response=f"{torch.cuda.is_available()}\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """
    """
    image = None

    if flask.request.content_type in ("image/png", "image/jpeg", "image/jpg"):
        image = ImageModule.string_to_image(flask.request.data)
    else:
        return flask.Response(
            response=f"This predictor only supports png, jpeg, jpg type images, but you sent {flask.request.content_type} type.", status=415, mimetype="text/plain"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = ImageModule.resize_with_padding(image, 904)
    answer_image = image.copy()
    mask = Image.new("RGB", (904, 904), (0, 0, 0))
    answers = create_mask(image)
    
    image_draw = ImageDraw.Draw(answer_image)
    mask_draw = ImageDraw.Draw(mask)
    for i, point in enumerate(answers):
        x, y, w, h = point
        image_draw.rectangle((x, y, x + w, y + h),  outline=(255, 0, 0), width=2)
        mask_draw.rectangle((x, y, x + w, y + h), fill=(255, 255, 255))

    image = transforms.ToTensor()(image).to(device).unsqueeze(0)
    mask = transforms.Grayscale()(transforms.ToTensor()(mask)).to(device).unsqueeze(0)
    mask = (mask > 0) * 1

    with torch.no_grad():
        inpainted_image = ImageModule.predict(image, mask)  

    cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

    is_success, output_buffer = cv2.imencode(".png", cur_res)
    if not is_success:
        return flask.Response(
            response="Failed to create an image.", status=415, mimetype="text/plain"
        )
        
    output_buffer = io.BytesIO(output_buffer)
    output_string = base64.b64encode(output_buffer.read()).decode('utf-8')

    payload = {'answers': answers, 'image': output_string}

    return flask.Response(response=json.dumps(payload), status=200, mimetype="text/png")
