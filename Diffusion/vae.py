import torch

import torch.nn as nn
import torch.nn.functional as F




class VAE(nn.Module):
	# feel free to define your arguments
	def __init__(self, hidden_dim=150, latent_dim=300, class_emb_dim=10, num_classes=10):
		super().__init__()
		
		self.encoder = nn.Linear(28*28, hidden_dim, True) # implement your encoder here

		# defining the network to estimate the mean
		self.mu_net = torch.nn.Linear(hidden_dim, latent_dim, True) # implement your mean estimation module here
		
		# defining the network to estimate the log-variance
		self.logvar_net = torch.nn.Linear(hidden_dim, latent_dim, True) # implement your log-variance estimation here
		
		# defining the class embedding module
		self.class_embedding = torch.nn.Embedding(num_classes, latent_dim) # implement your class-embedding module here

		# defining the decoder here
		self.decoder = nn.Sequential(nn.Linear(latent_dim,hidden_dim, True), nn.ReLU(),nn.Linear(hidden_dim,28*28, True),
		nn.Sigmoid())


	def forward(self, x: torch.Tensor, y: torch.Tensor):
		"""
		Args:
				x (torch.Tensor): image [B, 1, 28, 28]
				y (torch.Tensor): labels [B]
				
		Returns:
				reconstructed: image [B, 1, 28, 28]
				mu: [B, latent_dim]
				logvar: [B, latent_dim]
		"""
		


		x = x.view(x.size(0), -1)
		
		x1 = F.relu(self.encoder(x))

		mu = self.mu_net(x1) # mean 
		mu= F.relu(mu)
		mu = F.sigmoid(mu)
	
		logvar = self.logvar_net(x1) # log_var 
		logvar= F.relu(logvar)
		logvar = F.sigmoid(logvar)



		x2 = self.reparameterize(mu,logvar)

		#x3 = F.relu(self.fc2(x2))

		#labels=F.one_hot(y,num_classes=10)
		device = torch.device('cuda')

		#y=y.type(torch.float)
		#y=y.to(device)



		embeddings=self.class_embedding(y) # get embeddings 
		embeddings=embeddings+x2 # add embeddings to samples from reparameterization

		x4 = self.decoder(embeddings)
		reconstructed = x4.view(x.size(0), 1, 28, 28)


		return reconstructed, mu, logvar

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):

		device = torch.device('cuda')

		new_sample = mu + torch.randn(logvar.shape).to(device) * torch.exp(logvar/2.0) # sampling from normal distribution

		return new_sample

	def kl_loss(self, mu, logvar):
		"""
		getting the KL divergence between a normal distribution with mean "mu" and
		log-variance "logvar" and the standard normal distribution (mean=0, var=1)
		"""
		
		if torch.cuda.is_available():
			device=torch.device("cuda")
		else:
			device=torch.cuda('cpu')
			
		loss_tensor = logvar + mu**2 - torch.log(logvar) - 1.0
	
		kl_div = 0.5*(torch.sum(loss_tensor))# calculate the kl-div using mu and logvar

		return kl_div

	def get_loss(self, x: torch.Tensor, y: torch.Tensor):
		"""
		given the image x and the label y this function calculates the prior loss and reconstruction loss
		"""
		labels=y
		reconstructed, mu, logvar = self.forward(x, labels)

		# reconstruction loss
		# compute the reconstruction loss here using the "reconstructed" variable above
		device=torch.device('cuda')
		#recons_img=self.generate_sample(y.shape[0],y,device)
		loss=nn.L1Loss()
		recons_loss = loss(reconstructed,x)

		# prior matching loss
		prior_loss = self.kl_loss(mu, logvar)
		return recons_loss, prior_loss

	@torch.no_grad()
	def generate_sample(self, num_images: int, y, device):
		"""
		generates num_images samples by passing noise to the model's decoder
		if y is not None (e.g., y = torch.tensor([1, 2, 3]).to(device)) the model
		generates samples according to the specified labels
		
		Returns:
				samples: [num_images, 1, 28, 28]
		"""
		
		# sample from noise, find the class embedding and use both in the decoder to generate new samples
		
		
		
		labels=self.class_embedding(y)
		labels+=torch.randn((num_images,1),dtype=float).to(device)
		samples=self.decoder(labels)
	
		samples = samples.view((num_images,1,28,28))
		
		return samples


def load_vae_and_generate():
		device = torch.device('cuda')
		vae = VAE() # define your VAE model according to your implementation above
		
		# loading the weights of VAE
		vae.load_state_dict(torch.load('vae.pt'))
		vae = vae.to(device)
		
		desired_labels = []
		for i in range(10):
				for _ in range(5):
						desired_labels.append(i)

		desired_labels = torch.tensor(desired_labels,torch.float).to(device)
		generated_samples = vae.generate_sample(50, desired_labels, device)
		
		return generated_samples
