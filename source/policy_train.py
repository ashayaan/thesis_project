import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim


from policy_utils import DataProcessing
from policy_network import PolicyNetwork

from model_params import window_size
from model_params import epochs
from model_params import input_channels
from model_params import policy_learning_rate

class Train(nn.Module):
	def __init__(self,learning_rate,input_channels):
		super(Train,self).__init__()

		self.learning_rate = learning_rate
		self.reset()
		self.network = PolicyNetwork(input_channels)

	def reset(self):
		self.wealth = 10e4
		self.wealth_history = []
		self.loss = 0
		self.total_reward = 0

	def updateSummary(self,loss,reward):
		self.loss += loss
		self.wealth = self.wealth * np.exp(reward)
		self.total_reward += reward
		self.wealth_history.append(self.wealth)


if __name__ == '__main__':
	train = Train(policy_learning_rate,input_channels)
	print train.wealth

