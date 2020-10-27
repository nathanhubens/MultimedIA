import requests
from io import BytesIO

from PIL import Image
import numpy as np
import scipy.ndimage as nd
import torch

from google.colab import files

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1
from lucent.misc.io import show




@objectives.wrap_objective()
def dot_compare(layer, batch=1, cossim_pow=0):
  def inner(T):
    dot = (T(layer)[batch] * T(layer)[0]).sum()
    mag = torch.sqrt(torch.sum(T(layer)[0]**2))
    cossim = dot/(1e-6 + mag)
    return -dot * cossim ** cossim_pow
  return inner


def feature_inversion(img, layer=None, n_steps=512, cossim_pow=0.0, device='cpu'):
  # Convert image to torch.tensor and scale image
  img = torch.tensor(np.transpose(img, [2, 0, 1])).to(device)
  upsample = torch.nn.Upsample(224)
  img = upsample(img)
  
  obj = objectives.Objective.sum([
    1.0 * dot_compare(layer, cossim_pow=cossim_pow),
    objectives.blur_input_each_step(),
  ])

  # Initialize parameterized input and stack with target image
  # to be accessed in the objective function
  params, image_f = param.image(224)
  def stacked_param_f():
    return params, lambda: torch.stack([image_f()[0], img])

  transforms = [
    transform.pad(8, mode='constant', constant_value=.5),
    transform.jitter(8),
    transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1]*4),
    transform.random_rotate(list(range(-5, 5)) + [0]*5),
    transform.jitter(2),
  ]

  _ = render.render_vis(model, obj, stacked_param_f, transforms=transforms, thresholds=(n_steps,), show_image=False, progress=False)

  show(_[0][0])


def load(url):
    size=224,224
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize(size, Image.ANTIALIAS)
    return np.array(img, dtype='float32') / 255