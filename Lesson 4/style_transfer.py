import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1
from lucent.misc.io import show
from lucent.optvis.objectives import wrap_objective

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(url):
  size=512,512
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  img = img.resize(size, Image.ANTIALIAS)
  return np.array(img) / 255

def mean_L1(a, b):
  return torch.abs(a-b).mean()


def style_transfer_param(content_image, style_image, decorrelate=True, fft=True):
    shape = content_image.shape[:2] # assume we use content_image.shape
    params, image = param.image(*shape, decorrelate=decorrelate, fft=fft)
    def inner():
        style_transfer_input = image()[0]
        content_input = torch.tensor(np.transpose(content_image, [2, 0, 1])).float().to(device)
        style_input = torch.tensor(np.transpose(style_image[:shape[0], :shape[1], :], [2, 0, 1])).float().to(device)
        return torch.stack([style_transfer_input, content_input, style_input])
    return params, inner

# following the original Lucid notebook,
# these constants help remember which image is at which batch dimension
TRANSFER_INDEX = 0
CONTENT_INDEX = 1
STYLE_INDEX = 2


@wrap_objective()
def activation_difference(layer_names, activation_loss_f=mean_L1, transform_f=None, difference_to=CONTENT_INDEX):
    def inner(T):
        # first we collect the (constant) activations of image we're computing the difference to
        image_activations = [T(layer_name)[difference_to] for layer_name in layer_names]
        if transform_f is not None:
            image_activations = [transform_f(act) for act in image_activations]

        # we also set get the activations of the optimized image which will change during optimization
        optimization_activations = [T(layer)[TRANSFER_INDEX] for layer in layer_names]
        if transform_f is not None:
            optimization_activations = [transform_f(act) for act in optimization_activations]

        # we use the supplied loss function to compute the actual losses
        losses = [activation_loss_f(a, b) for a, b in zip(image_activations, optimization_activations)]
        return sum(losses)

    return inner



def gram_matrix(features, normalize=True):
    C, H, W = features.shape
    features = features.view(C, -1)
    gram = torch.matmul(features, torch.transpose(features, 0, 1))
    if normalize:
        gram = gram / (H * W)
    return gram



def style_transfer(content_image, style_image, model, content_layers, style_layers, content_weight=500, style_weight=1):

  param_f = lambda: style_transfer_param(content_image, style_image)

  content_obj = activation_difference(content_layers, difference_to=CONTENT_INDEX)
  content_obj.description = "Content Loss"

  style_obj = activation_difference(style_layers, transform_f=gram_matrix, difference_to=STYLE_INDEX)
  style_obj.description = "Style Loss"

  objective = content_weight * content_obj + style_weight * style_obj

  vis = render.render_vis(model, objective, param_f, show_inline=True)