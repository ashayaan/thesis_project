import torch
import torchvision
import visdom
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
import math


from policy_utils import DataProcessing
from policy_utils import Plotter
from policy_network import PolicyNetwork


from model_params import window_size
from model_params import epochs
from model_params import input_channels
from model_params import policy_learning_rate
from model_params import train_size
from model_params import policy_batch_size as batch_size
from model_params import transaction_commission

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
		self.loss = 0
		self.total_reward = 0
		self.wealth_history = []

	def updateSummary(self,loss):
		self.loss += loss
		reward = -1 * loss
		self.wealth = self.wealth * math.exp(reward)
		self.total_reward += reward
		self.wealth_history.append(self.wealth)

	def getBatch(self,data,target,batch):
		start_pos = batch * self.batch_size
		end_pos = start_pos + self.batch_size
		x = torch.from_numpy(data[start_pos:end_pos]).float()
		y = torch.from_numpy(target[start_pos:end_pos]).float()
		# x = x.view(x.shape[0],x.shape[1]*x.shape[2])
		# y = y.view(y.shape[0],1)
		
		#Normalizing X by the last real day price
		x = x / x[-1][-1][-2]
		return x,y

	def loss_function(self,w1,w2,y):
		transaction_cost = 1 - torch.sum(torch.abs(w2-w1))*transaction_commission
		portfolio_return = torch.sum(w2*y)
		loss = -1 * torch.log(portfolio_return*transaction_cost)
		loss = torch.mean(loss)
		return loss

def trainNetwork(net,train_data,train_target,iterations):
	num_batches = (len(train_data)//net.batch_size)
	for batch in range(num_batches):
		iterations+=1

		x,y = net.getBatch(train_data,train_target,batch)
		
		#Forward Propagation
		previous_weights = net.network.weight_buffer[-1]
		new_weights = net.network.forward(x,previous_weights)

		#Calculating Loss
		loss = net.loss_function(previous_weights,new_weights,y)
		
		#Back Propagation
		net.optimizer.zero_grad()
		loss.backward(retain_graph=True)
		net.optimizer.step()

		# print net.wealth*torch.exp(-1*loss.item())
		net.updateSummary(loss.item())
		plotter.plot('Wealth', 'iterations', 'Policy Wealth', iterations, net.wealth)

	print('Wealth:{} Loss:{}').format(net.wealth,net.loss/num_batches)
	#TODO update summary 
	return net,iterations

if __name__ == '__main__':
	global plotter
	plotter = Plotter(env_name='Policy Network')

	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="../data", help="path to the dataset")
	parser.add_argument("--file", type=str, default="combined.csv")
	parser.add_argument("--models", type=str, default="../saved_models", help="path to the dataset")
	args = parser.parse_args()
	file_name = args.datapath + '/' + args.file


	net = Train(policy_learning_rate,input_channels)
	
	# print net.network.weight_buffer

	data = DataProcessing(file_name,train_size,args.models)
	train_data, train_target = data.trainingData()

	iterations = 0
	for epoch in range(net.epochs):
		print '--------------------------'
		print('Epoch: {}').format(epoch+1)
		print '-------------------------'
		net,iterations = trainNetwork(net,train_data,train_target,iterations)
		net.reset()
		net.network.resetBuffer()


