import torch
import torchvision.models as models
from packnet_sfm.networks.depth.DepthResNet import DepthResNet

from packnet_sfm.networks.layers.resnet.resnet_encoder import ResnetEncoder

from packnet_sfm.networks.depth.DepthResNetSwin import DepthResNetSwin

from packnet_sfm.networks.depth.DepthResNetCMT import DepthResNetCMT

from ptflops import get_model_complexity_info

with torch.cuda.device(0):

  #net = DepthResNet(version="18pt")
  #net = DepthResNetSwin(version="18pt")

  #net = DepthResNetCMT(version="50pt")


  macs, params = get_model_complexity_info(net, (3, 192, 640), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))