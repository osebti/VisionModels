import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple



def sinusoidal_embedding(n, d):
  # Returns the standard positional embedding
  embedding = torch.zeros(n, d)
  wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
  wk = wk.reshape((1, d))
  t = torch.arange(n).reshape((n, 1))
  embedding[:,::2] = torch.sin(t * wk[:,::2])
  embedding[:,1::2] = torch.cos(t * wk[:,::2])

  return embedding



class MyBlock(nn.Module):
  def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
    super(MyBlock, self).__init__()
    self.ln = nn.LayerNorm(shape)
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
    self.activation = nn.SiLU() if activation is None else activation
    self.normalize = normalize

  def forward(self, x):
    out = self.ln(x) if self.normalize else x
    out = self.conv1(out)
    out = self.activation(out)
    out = self.conv2(out)
    out = self.activation(out)
    return out



class VarianceScheduler:
  """
  This class is used to keep track of statistical variables used in the diffusion model
  and also adding noise to the data
  """
  def __init__(self, beta_start: float=0.0001, beta_end: float=0.02, num_steps :int=1000):
    device= "cuda" if torch.cuda.is_available() else "cpu"
    self.betas = torch.linspace(beta_start, beta_end, num_steps)  # Number of steps is typically in the order of thousands
    self.alphas = 1 - self.betas
    self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])
    self.steps=num_steps
    


  
  
  def add_noise(self, x: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This method receives the input data and the timestep, generates a noise according to the 
    timestep, perturbs the data with the noise, and returns the noisy version of the data and
    the noise itself
    
    Args:
      x (torch.Tensor): input image [B, 1, 28, 28]
      timestep (torch.Tensor): timesteps [B]

    Returns:
      Tuple[torch.Tensor, torch.Tensor]: noisy_x [B, 1, 28, 28], noise [B, 1, 28, 28]
    """

    device="cuda" if torch.cuda.is_available() else "cpu"
    eta = torch.randn(x.shape)
    noisy_x=torch.zeros(x.shape)
    n=int(x.shape[0])

    if len(timestep.shape)==0:
      timestep=torch.tensor([timestep]).type(torch.int64)
    
    x=x.cpu()
    
    for i in range(n):
      a_bar = self.alpha_bars[timestep[i]]
      noisy_x[i] = a_bar.sqrt().reshape(1, 1, 1, 1) * x[i] + (1 - a_bar).sqrt().reshape(1, 1, 1, 1) * eta[i]


    return noisy_x, eta
    


class NoiseEstimatingNet(nn.Module):
  """
  The implementation of the noise estimating network for the diffusion model
  """
  # feel free to add as many arguments as you need or change the arguments
  def __init__(self, time_emb_dim: int=64, num_classes: int=10, n_steps=1000):
  
    super(NoiseEstimatingNet, self).__init__()

    # Sinusoidal embedding
    
    self.time_embed = nn.Embedding(n_steps, time_emb_dim)
    self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
    self.time_embed.requires_grad_(False)
    self.class_embed = nn.Embedding(num_classes,time_emb_dim)
    self.steps=n_steps
    #self.time_embed.requires_grad_(False)

    # First half
    self.te1 = self._make_te(time_emb_dim, 1)
    self.b1 = nn.Sequential(
      MyBlock((1, 28, 28), 1, 10),
      MyBlock((10, 28, 28), 10, 10),
      MyBlock((10, 28, 28), 10, 10)
    )
    self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

    self.te2 = self._make_te(time_emb_dim, 10)
    self.b2 = nn.Sequential(
      MyBlock((10, 14, 14), 10, 20),
      MyBlock((20, 14, 14), 20, 20),
      MyBlock((20, 14, 14), 20, 20)
    )
    self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

    self.te3 = self._make_te(time_emb_dim, 20)
    self.b3 = nn.Sequential(
      MyBlock((20, 7, 7), 20, 40),
      MyBlock((40, 7, 7), 40, 40),
      MyBlock((40, 7, 7), 40, 40)
    )
    self.down3 = nn.Sequential(
      nn.Conv2d(40, 40, 2, 1),
      nn.SiLU(),
      nn.Conv2d(40, 40, 4, 2, 1)
    )

    # Bottleneck
    self.te_mid = self._make_te(time_emb_dim, 40)
    self.b_mid = nn.Sequential(
      MyBlock((40, 3, 3), 40, 20),
      MyBlock((20, 3, 3), 20, 20),
      MyBlock((20, 3, 3), 20, 40)
    )

    # Second half
    self.up1 = nn.Sequential(
      nn.ConvTranspose2d(40, 40, 4, 2, 1),
      nn.SiLU(),
      nn.ConvTranspose2d(40, 40, 2, 1)
    )

    self.te4 = self._make_te(time_emb_dim, 80)
    self.b4 = nn.Sequential(
      MyBlock((80, 7, 7), 80, 40),
      MyBlock((40, 7, 7), 40, 20),
      MyBlock((20, 7, 7), 20, 20)
    )

    self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
    self.te5 = self._make_te(time_emb_dim, 40)
    self.b5 = nn.Sequential(
      MyBlock((40, 14, 14), 40, 20),
      MyBlock((20, 14, 14), 20, 10),
      MyBlock((10, 14, 14), 10, 10)
    )

    self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
    self.te_out = self._make_te(time_emb_dim, 20)
    self.b_out = nn.Sequential(
      MyBlock((20, 28, 28), 20, 10),
      MyBlock((10, 28, 28), 10, 10),
      MyBlock((10, 28, 28), 10, 10, normalize=False)
    )

    self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

  def forward(self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  
    # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
    t = self.time_embed(timestep)
    c= self.class_embed(y.unsqueeze(1))
    t=t+c
    n = len(x)
    out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
    out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
    out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

    out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

    out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
    out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

    out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
    out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

    out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
    out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

    out = self.conv_out(out)

    return out
  
  def _make_te(self, dim_in, dim_out):
    return nn.Sequential(
      nn.Linear(dim_in, dim_out),
      nn.SiLU(),
      nn.Linear(dim_out, dim_out)
    )



class DiffusionModel(nn.Module):
  """
  The whole diffusion model put together
  """
  def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler):

    super().__init__()
    
    self.network = network # UNet
    self.var_scheduler = var_scheduler
  
  def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.float32:

    
    # step1: sample timesteps
    # step2: compute the noisy versions of the input image according to your timesteps
    # step3: estimate the noises using the noise estimating network
    # step4: compute the loss between the estimated noises and the true noises

    device="cuda" if torch.cuda.is_available() else "cpu"
    timesteps=torch.randint(0, self.var_scheduler.steps,(x.shape[0],1),dtype=torch.int64) # sample timesteps
    
    noise_gt,eta=self.var_scheduler.add_noise(x,timesteps.cpu())
    x=x.to(device)
    y=y.to(device)
    timesteps=timesteps.to(device)
    noise_gt=noise_gt.to(device)
    noise_pred=self.network.forward(noise_gt,timesteps,y)

    mse = nn.MSELoss()
    loss=mse(noise_pred.to(device),eta.to(device))
    
    return loss
  
  @torch.no_grad()
  def generate_sample(self, num_images: int, y, device) -> torch.Tensor:
    """
    This method generates as many samples as specified according to the given labels
    
    Args:
    num_images (int): number of images to generate
    y (_type_): the corresponding expected labels of each image
    device (_type_): computation device (e.g. torch.device('cuda')) 

    Returns:
    torch.Tensor: the generated images [num_images, 1, 28, 28]
    """

    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    #timesteps = torch.linspace(0, 200,steps=200,dtype=torch.int64)

    with torch.no_grad():
          device=device

      # Starting from random noise
          x = torch.randn(num_images, 1, 28, 28).to(device)
          
          for t in range(self.var_scheduler.steps-1,0,-1):
            # Estimating noise to be removed
            time_tensor = (torch.ones((num_images,1)) * t).type(torch.int64).to(device)
            eta_theta = self.network.forward(x, time_tensor,y)
            

            alpha_t = self.var_scheduler.alphas[t]
            beta=self.var_scheduler.betas[t]
            alpha_t_bar = self.var_scheduler.alpha_bars[t]
            z = torch.randn(num_images, 1, 28, 28).to(device) #noise

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (beta * eta_theta / (1 - alpha_t_bar).sqrt())) + beta.sqrt()*z

         
          return x



def load_diffusion_and_generate():
  device = torch.device('cuda')
  var_scheduler = VarianceScheduler() # define your variance scheduler
  network = NoiseEstimatingNet() # define your noise estimating network
  diffusion = DiffusionModel(network=network, var_scheduler=var_scheduler) # define your diffusion model
  
  # loading the weights of VAE
  diffusion.load_state_dict(torch.load('diffusion.pt'))
  diffusion = diffusion.to(device)
  
  desired_labels = []
  for i in range(10):
    for _ in range(5):
      desired_labels.append(i)

  desired_labels = torch.tensor(desired_labels).to(device)
  generated_samples = diffusion.generate_sample(50, desired_labels, device)
  
  return generated_samples
