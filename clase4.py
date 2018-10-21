import unittest
import random
import math
import numpy as np

import matplotlib.pyplot as plt

class SigmoidNeuron:

	def __init__(self,w=[],b=0):
		self.weights=w
		self.bias=b

	def mrand(self,n_input):
		w = []
		for i in range(0, n_input):
			w.append(random.uniform(-2.0, 2.0))
		self.weights = w
		self.bias = random.uniform(-2.0, 2.0)


	def feed(self,x):
		summ = 0.0
		for i,w in enumerate(self.weights):
			summ+= w*x[i]
		return 1.0 / (1.0 + np.exp(-summ -self.bias) )

	def upgrade_wb(self, delta, input, rate):
		for j, w in enumerate(self.weights):
			self.weights[j] += (rate * delta * input[j])
		self.bias += (rate * delta)


class NeuronLayer:
	def __init__(self,n_input,n_neurons):
		neurons =[]
		for i in range(0,n_neurons):
			sn = SigmoidNeuron()
			sn.mrand(n_input)
			neurons.append(sn)
		self.neurons=neurons

	def feed(self,x):
		res = []
		for n in self.neurons:
			res.append(n.feed(x))
		return res
	'''
	def error_backpropagation(self,delta,output,nweight):
		ndelta=[]
		for i,n in enumerate(self.neurons):
			ne=0.0
			for j,w in enumerate(nweight):
				ne+= nweight[j][i]*delta[j]
			nd=ne*(output[i] * (1.0 - output[i]))			# o=np.multiply(output,(1.0 -output), dtype=np.float64)
			ndelta.append(nd)
		return ndelta
	'''

	def upgrade_wb(self, delta, input, rate):
		for i,n in enumerate(self.neurons):
			self.neurons[i].upgrade_wb(delta[i], input, rate)
		return self.feed(input)


	def get_weight(self):
		res = []
		for n in self.neurons:
			res.append(n.weights)
		return res

	def get_bias(self):
		res = []
		for n in self.neurons:
			res.append(n.bias)
		return res




class NeuralNetwork:

	def __init__(self,layers):
		self.layers=layers

	def forward_feeding(self,input):
		res = outputs =[]
		x = input
		for i,l in enumerate(self.layers):
			res = l.feed(x)
			outputs.append(res)
			x = res
		return res, outputs

	'''
	def error_backpropagation2(self, outputs, expected_output):
		n_layers = self.layers.__len__()
		output = outputs[outputs.__len__()-1]
		output_layer = self.layers[n_layers-1]
		error=[]
		delta=[]
		deltam=[]
		for i,n in enumerate(output_layer.neurons):
			e=(expected_output[i] - output[i])
			d = e*(output[i] * (1.0 - output[i]))
			error.append(e)
			delta.append(d)
		deltam.append(delta)
		for i in range(2, n_layers):
			print(n_layers - i)
			l = self.layers[n_layers - i]
			nl = self.layers[n_layers - i + 1]
			loutput = outputs[n_layers - i]
			nweight = nl.get_weight()
			delta = l.error_backpropagation(delta, loutput, nweight)
			deltam.append(delta)
		return deltam[::-1]
	'''

	def error_backpropagation(self, outputs, expected_output):

		n_layers = self.layers.__len__()
		output = outputs[outputs.__len__()-1]
		output_layer = self.layers[n_layers-1]

		error=[]
		delta=[]
		deltam=[]

		#print("outputs",outputs)
		#print("weights",self.get_weight())

		for i,n in enumerate(output_layer.neurons):
			e=(expected_output[i] - output[i])
			d = e*(output[i] * (1.0 - output[i]))
			error.append(e)
			delta.append(d)

		deltam.append(delta)
		#print("o ",n_layers-1)
		#print(self.get_weight())

		for i in range(2,n_layers+1):
			il = n_layers -i
			inl = n_layers -i +1
			#print("l ",il,"\tnl",inl)

			ndelta= delta
			delta = []
			l = self.layers[il]
			nl = self.layers[inl]
			loutput = outputs[il]
			nweight = nl.get_weight()

			for i, n in enumerate(l.neurons):
				#print(nweight,ndelta)
				e = 0.0
				for j,w in enumerate(nweight) :
					for w in nweight[j]:
						e += w * ndelta[j]
				d = e * (loutput[i] * (1.0 - loutput[i]))
				error.append(e)
				delta.append(d)
			deltam.append(delta)

		return deltam[::-1]


	def upgrade_wb(self, deltam, input, learn_rate, outputs):
		for i,l in enumerate(self.layers):
			l.upgrade_wb(deltam[i], input, learn_rate)
			input = outputs[i]


	def get_weight(self):
		res = []
		for l in self.layers:
			res.append(l.get_weight())
		return res

	def get_bias(self):
		res = []
		for l in self.layers:
			res.append(l.get_bias())
		return res







def realVal_2DLine(n_input):
	inputs = []
	results = []
	for i in range(0,n_input):
		xx= random.uniform(-50.0, 50.0)+0.0
		xy= random.uniform(-50.0, 50.0)

		inputs.append([xx,xy])
		results.append(0.0+xx<xy)

	return inputs,results

def plot_deep_2DLine(nn,n_input,x,title):
	fig, ax = plt.subplots()
	color = ['red', 'blue']

	for j in range(0, n_input):
		res, outputs = nn.forward_feeding(x[j])
		#print(x[j],"\t",res)
		c = res[0] > 0.5
		#print("xj",x[j])
		#print("c ",c)
		xx = x[j][0]
		yy = x[j][1]
		scale = 20
		ax.scatter(xx, yy, c=color[c], s=scale, alpha=0.5, edgecolors='none')

	plt.title(title)
	plt.show()

'''
#layer(size/n input x neu, n neurons)
def learn_deep():
	size = 100
	trainings= 10000
	learn_rate = 0.5
	layers = [

		NeuronLayer(2, 1),
	]
	nn = NeuralNetwork(layers)

	success = 0.0

	for i in range(0,trainings):

		x,dOutput = realVal_2DLine(size)
		if (i % 100 == 0):
			ratio= int(100*(success/(size+0.0)))
			plot_deep_2DLine(nn, size, x, i.__str__() + " trainings / "+ratio.__str__()+"%")
		success=0.0

		for j in range(0, size):

			res,outputs = nn.forward_feeding(x[j])
			deltam=nn.error_backpropagation(outputs,dOutput)
			nn.upgrade_wb(deltam,x[j],learn_rate,outputs)

			diff = dOutput[j]-res[0]
			success += (1.0 - abs(diff))


		#print(nn.layers[0].neurons[0].bias)
		#print(nn.layers[0].neurons[0].weights[0])
		if (i % 50 == 0):
			ratio= int(100*(success/(size+0.0)))
			print(i,"\t",ratio,"%")

'''

def learn_or():
	size = 4
	trainings= 1000
	learn_rate = 0.4
	layers = [
		NeuronLayer(2, 2),
		NeuronLayer(2, 2),
		NeuronLayer(2, 1),
	]
	nn = NeuralNetwork(layers)

	for t in range(0,trainings):
		success = 0.0

		for i in range(0,2):
			for j in range(0,2):
				x=[i,j]
				real = i or j

				res, outputs = nn.forward_feeding(x)
				#print(outputs)
				deltam = nn.error_backpropagation(outputs, [real])
				nn.upgrade_wb(deltam, x, learn_rate,outputs)

				result = res[0] > 0.5
				success += (1.0 - abs(real - result))

				#print(nn.layers[0].neurons[0].bias)
				#print(nn.layers[0].neurons[0].weights[0])

		if (t % 10 == 0):
			ratio= (success/(size+0.0))
			print(t,"\t",ratio)

	return nn


def learn_line():
	size = 4
	trainings= 1000
	learn_rate = 0.4
	layers = [
		NeuronLayer(2, 5),
		NeuronLayer(5, 2),
		NeuronLayer(2, 1),
	]
	nn = NeuralNetwork(layers)

	success = 0.0
	for t in range(0,trainings):
		xx= random.uniform(-50.0, 50.0)+0.0
		xy= random.uniform(-50.0, 50.0)

		x = [xx,xy]
		real = 0.0+xx<xy

		res, outputs = nn.forward_feeding(x)
		deltam = nn.error_backpropagation(outputs, [real])
		nn.upgrade_wb(deltam, x, learn_rate, outputs)

		success += (1.0 - abs(real - res[0]))
		if (t % 100 == 0):
			ratio= (success/100)
			success=0.0
			print(t,"\t",ratio)

	return nn

def test_line():
	nn =learn_line()
	print("weights:\t", nn.get_weight())
	print("bias:   \t", nn.get_bias())

	size=100
	x, dOutput = realVal_2DLine(size)
	plot_deep_2DLine(nn, size, x, "%")


def test_or():
	nn = learn_or()
	print("weights:\t", nn.get_weight())
	print("bias:   \t", nn.get_bias())
	for i in range(0,2):
		for j in range(0,2):
			x=[i,j]
			real = i or j
			res, outputs = nn.forward_feeding(x)
			result = res[0] > 0.5
			print(x,"\treal:",real,"\tres:",result)


test_or()
test_line()
#learn_deep()
