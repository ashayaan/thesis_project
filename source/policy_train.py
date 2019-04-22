import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse



from policy_utils import DataProcessing
from policy_network import PolicyNetwork

from model_params import window_size
from model_params import epochs
from model_params import input_channels
from model_params import policy_learning_rate
from model_params import train_size
from model_params import policy_batch_size as batch_size

class Train(nn.Module):
	def __init__(self,learning_rate,input_channels):
		super(Train,self).__init__()

		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.reset()
		self.network = PolicyNetwork(input_channels)
		self.parameters = self.network.parameters()
		self.optimizer = optim.Adam(self.parameters,lr=self.learning_rate)


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

	def getBatch(self,data,target,batch):
		start_pos = batch * self.batch_size
		end_pos = start_pos + self.batch_size
		x = torch.from_numpy(data[start_pos:end_pos]).float()
		y = torch.from_numpy(target[start_pos:end_pos]).float()
		# x = x.view(x.shape[0],x.shape[1]*x.shape[2])
		y = y.view(y.shape[0],1)
		return x,y

	def loss(w1,w2):
		pass

def trainNetwork(net,train_data,train_target):
	num_batches = (len(data)//net.batch_size)
	for batch in range(num_batches):

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="../data", help="path to the dataset")
	parser.add_argument("--file", type=str, default="combined.csv")
	parser.add_argument("--models", type=str, default="../saved_models", help="path to the dataset")
	args = parser.parse_args()
	file_name = args.datapath + '/' + args.file


	net = Train(policy_learning_rate,input_channels)
	data = DataProcessing(file_name,train_size,args.models)
s	train_data, train_target = data.trainingData()

	train_data = []
	train_target = []

	for epoch in range(net.epochs):
		net,reward = trainNetwork(net,train_data,train_target)

