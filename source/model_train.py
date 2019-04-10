import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import argparser

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
		self.optimizer = optim.Adam(lr = self.learning_rate)


	#Calculating the loss for the output
	def loss(self,data,out):
		mean = out[:,:15]
		std = out[:,15:]
		
		loss = (data-mean**2 / (2*std)) + torch.log(std)
		loss = torch.sum(loss)
		loss = loss/self.input_size

		return loss


def readData(file):
	df = pd.read_csv(file)
	df.drop(columns = ['Date'], inplace = True)
	return df

def trainNetwork(data, net):
	pass

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
	