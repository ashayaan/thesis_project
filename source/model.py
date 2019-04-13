import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd


from model_params import input_size

class Network(nn.Module):
	def __init__(self,input_size):
		super(Network, self).__init__()
		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.fc1 = nn.Linear(input_size, 20)
		self.fc2 = nn.Linear(20, 25)
		self.fc3 = nn.Linear(25, 28)
		# self.fc4 = nn.Linear(25,28)

	def forward(self,data):
		out = self.relu(self.fc1(data))
		out = self.relu(self.fc2(out))
		out = self.relu(self.fc3(out))
		# out = self.relu(self.fc4(out))
		return out


'''Testing network'''
if __name__ == '__main__':
	x = torch.randn((1,14))
	net = Network(input_size)

	print net.forward(x).shape