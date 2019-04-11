import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import argparse
from torch.utils.data import Dataset, DataLoader

from model import Network
from model_params import batch_size
from model_params import input_size
from model_params import learning_rate
from model_params import num_epochs

class Train(nn.Module):
	def __init__(self):
		super(Train,self).__init__()

		self.input_size = input_size
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.network = Network(self.input_size)

		self.parameters = self.network.parameters()
		self.optimizer = optim.Adam(self.parameters, lr = self.learning_rate)


	#Calculating the loss for the output
	def loss(self,data,out):
		mean = out[:,:14]
		std = out[:,14:]
		print torch.log(std)
		loss = ((data-mean)**2 / 2*(std**2)) + torch.log(std)
		loss = torch.sum(loss)
		loss = loss/self.input_size

		return loss

'''
Reads and normalizes the data
Normalization scheme min max
'''
def readData(file):
	df = pd.read_csv(file)
	df.drop(columns = ['Date'], inplace = True)
	df = df.apply(lambda x: (x-np.min(x))/(np.max(x) - np.min(x)))
	df = df.values.astype(dtype=np.float32())
	return df

def get_batch(data,batch,net):
	return torch.from_numpy(data[batch : batch + net.batch_size])

def trainNetwork(data, net):
	# n_batches = len(data)//net.batch_size
	for batch in range(len(data)-net.batch_size):
		X = get_batch(data,batch,net)
		output = net.network.forward(X)
		loss = net.loss(X,output)
		

		net.optimizer.zero_grad()
		loss.backward()
		net.optimizer.step()
		# print('LOSS: {}'.format(loss.item(s)))
	
	return net

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--datapath", type=str, default="../data", help="path to the dataset")
	parser.add_argument("--file", type=str, default="combined.csv")
	args = parser.parse_args()

	file_name = args.datapath + '/' + args.file

	data = readData(file_name)
	net = Train()

	for epoch in range(num_epochs):
		net = trainNetwork(data,net)
	