import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import visdom
import argparse
import pandas as pd
import numpy as np

from model_params import input_size
from model_params import sequence_size
from model_params import train_size
from model_params import learning_rate
from model_params import num_epochs
from model_params import batch_size

from model import Network
from utils import Plotter
from utils import DataProcessing

class Train(nn.Module):
	def __init__(self):
		super(Train,self).__init__()
		self.input_size = input_size
		self.batch_size = batch_size
		self.sequence_size = sequence_size
		self.learning_rate = learning_rate
		'''Constructing the network'''
		self.network = Network(self.input_size)
		self.loss_function = nn.MSELoss(reduction='mean')
		'''Defining the optimizer'''
		self.parameters = self.network.parameters()
		self.optimizer = optim.Adam(self.parameters, lr = self.learning_rate)


	'''Calculating the loss for the network'''
	def loss(self,predicted,target):
		return self.loss_function(predicted,target)

'''getting batches and flattening the input tensor'''
def getBatch(net,data,target,batch):
	x = torch.from_numpy(data[batch:batch+net.batch_size]).float()
	y = torch.from_numpy(target[batch:batch+net.batch_size]).float()
	x = x.view(x.shape[0],x.shape[1]*x.shape[2])
	return x,y

def trainNetwork(net,data,target,num_iterations,epoch):
	
	for batch in range(len(data)//net.batch_size):
	# for batch in range(2):
		num_iterations+=1
		X,Y = getBatch(net,data,target,batch)
		out = net.network.forward(X)
		loss = net.loss_function(out,Y)
		
		net.optimizer.zero_grad()
		loss.backward()
		net.optimizer.step()

		print('Epoch: {} Iteration:{} LOSS: {}'.format(epoch, num_iterations, loss.item()))
		plotter.plot('Loss', 'Train', 'Training Loss', num_iterations, loss.detach().numpy())
	
	return net,num_iterations
	


if __name__ == '__main__':
	num_iterations = 0
	global plotter
	plotter = Plotter(env_name='Thesis Project')

	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="../data", help="path to the dataset")
	parser.add_argument("--file", type=str, default="combined.csv")
	args = parser.parse_args()
	file_name = args.datapath + '/' + args.file

	data = DataProcessing(file_name,train_size)
	train_data,train_target = data.trainingData()

	net = Train()

	for epoch in range(num_epochs):
		net,num_iterations = trainNetwork(net,train_data,train_target,num_iterations,epoch)

	torch.save(net.network,'../saved_models/model.pt')
