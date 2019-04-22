import torch
import numpy as np
import pandas as pd

from model_params import window_size
from model import Network
from model_params import input_size

class DataProcessing(object):
	def __init__(self,file_name,train_size,path):
		self.df = pd.read_csv(file_name)
		self.df = self.df.drop(columns=['Date'])
		self.train_size = train_size
		self.index = int(len(self.df) * self.train_size)
		self.window = window_size
		self.train_data = self.df.iloc[:self.index]
		self.test_data = self.df.iloc[self.index:]

		self.trained_models = self.getModels(path)

	def getModels(self,path):
		name = ['SBI','LUPIN','CIPLA','KOTAK','LARSON','TATASTEEL','WIPRO','BOSCH','JINDAL','SUN','BHEL','HDFC','INFY','TATAPOWER']
		models = []
		for n in name:
			model_name = str(path)+'/model_'+n+'.pt'
			temp = Network(input_size)
			print model_name
			temp.load_state_dict(torch.load(model_name))
			temp.eval()
			models.append(temp)		
		return models

	def trainingData(self):
		print "---------------------------------------------"
		print "Loading Data"
		print "---------------------------------------------\n"
		attributes = []
		target = []
		#Creating Window
		for i in range( (len(self.train_data)//self.window)*self.window - self.window - 1 ):
			x = np.array(self.train_data.iloc[i:i+self.window],np.float64)
			y = np.array(self.train_data.iloc[i+self.window],np.float64)
			x = x.T
			predicted = [] 
			#Predicting next and concatinnating
			for	j in range(len(x)):
				normalization_factor = 1
				if j in[0,1,2,3,6,8,9,10,12,13]:
					normalization_factor = 10
				elif j in [4,5,11]:
					normalization_factor = 50
				else:
					normalization_factor = 250

				input_data = torch.from_numpy(x[j]/normalization_factor).float()
				pred = self.trained_models[j].forward(input_data) * normalization_factor
				predicted.append(pred.item())
			
			x = np.concatenate((x,np.array(predicted).reshape(14,1)),axis=1).reshape(1,14,6)

			attributes.append(x)
			target.append(y)
		
		return np.array(attributes),np.array(target)

'''Testing utils for policy network'''
if __name__ == '__main__':
	x = DataProcessing('../data/combined.csv',0.8,'../saved_models')
	attributes,target = x.trainingData()

	print attributes[0]
	print attributes[0].shape
	print target
