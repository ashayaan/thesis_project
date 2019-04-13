import torch
import numpy as np
import pandas as pd
import numpy as np


from visdom import Visdom
from model_params import train_size
from model_params import sequence_size

class Plotter(object):
	"""Plots to Visdom"""
	def __init__(self, env_name='main'):
		self.viz = Visdom()
		self.env = env_name
		self.plots = {}
	def plot(self, var_name, split_name, title_name, x, y):
		if var_name not in self.plots:
			self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
				legend=[split_name],
				title=title_name,
				xlabel='Iterations',
				ylabel=var_name
			))
		else:
			self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')



class DataProcessing(object):
	'''Helper class to generate train and test files'''
	def __init__(self,file_name):
		self.df = pd.read_csv(file_name)
		self.df = self.df.drop(columns=['Date'])
		self.train_size = train_size
		self.index = int(len(self.df) * self.train_size)
		self.seq_size = sequence_size

		self.train_data = self.df[:self.index]
		self.test_data = self.df[self.index:]

	def trainingData(self):
		return self.train_data

	def testingData(self):
		return self.test_data

if __name__ == '__main__':
	test = DataProcessing('../data/combined.csv')