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


	def error_backpropagation(self, outputs, expected_output):
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

		for i in range(2,n_layers+1):
			il = n_layers -i
			inl = n_layers -i +1

			ndelta= delta
			delta = []
			l = self.layers[il]
			nl = self.layers[inl]
			loutput = outputs[il]
			nweight = nl.get_weight()

			for i, n in enumerate(l.neurons):
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






def real_xor(n_input):
	inputs = []
	results = []
	for i in range(0,n_input):
		xx= random.randint(0,1)
		xy= random.randint(0,1)

		inputs.append([xx,xy])
		results.append(np.logical_xor(xx,xy))

	return inputs,results
def plot_nn(nn,n_input,x,title):
	fig, ax = plt.subplots()
	color = ['red', 'blue']

	for j in range(0, n_input):
		res, outputs = nn.forward_feeding(x[j])
		c = res[0] > 0.5
		xx = x[j][0]
		yy = x[j][1]
		scale = 20
		ax.scatter(xx, yy, c=color[c], s=scale, alpha=0.5, edgecolors='none')

	plt.title(title)
	plt.show()

def realVal_2DLine(n_input):
	inputs = []
	results = []
	for i in range(0,n_input):
		xx= random.uniform(-50.0, 50.0)+0.0
		xy= random.uniform(-50.0, 50.0)

		inputs.append([xx,xy])
		results.append(0.0+xx<xy)

	return inputs,results

def make_layers(nneurons):
	layers = []
	for i in range(0,nneurons.__len__()-1):
		layers.append(NeuronLayer(nneurons[i], nneurons[i+1]))
	return layers

'''
make_layers([2,4,2,1])
layers = [
	NeuronLayer(2, 4), # 2 input, # 4 neuron
	NeuronLayer(4, 2), # 4 input, # 2 neuron
	NeuronLayer(2, 1), # 2 input, # 1 neuron
]
'''

def learn_or():
	size = 4
	trainings= 1000
	learn_rate = 0.4
	layers = make_layers([2,4,2,1])

	nn = NeuralNetwork(layers)

	for t in range(0,trainings):
		success = 0.0

		for i in range(0,2):
			for j in range(0,2):

				x=[i,j]
				real = i or j
				res, outputs = nn.forward_feeding(x)

				deltam = nn.error_backpropagation(outputs, [real])
				nn.upgrade_wb(deltam, x, learn_rate,outputs)

				result = res[0] > 0.5
				success += (1.0 - abs(real - result))

		if (t % 10 == 0):
			ratio= (success/(size+0.0))
			print(t,"\t",ratio)
	return nn


def learn_xor():
	size = 4
	trainings= 1000
	learn_rate = 0.4
	layers = make_layers([2,10,1])

	nn = NeuralNetwork(layers)

	for t in range(0,trainings):
		success = 0.0
		for s in range(0,size):
			i = random.randint(0,1)
			j = random.randint(0,1)
			x=[i,j]
			real = np.logical_xor(i,j)+0.0

			res, outputs = nn.forward_feeding(x)
			deltam = nn.error_backpropagation(outputs, [real])
			nn.upgrade_wb(deltam, x, learn_rate,outputs)

			result = res[0] > 0.5
			success += (1.0 - abs(real - result))


		if (t % 10 == 0):
			ratio= (success/(size+0.0))
			print(t,"\t",ratio)

	return nn


def logical_inputs(size):
	inputs = []
	for i in range(0,size):
		xx= random.randint(0,1)
		xy= random.randint(0,1)
		inputs.append([xx,xy])
	return inputs

def learn_line():
	size = 4
	trainings= 1000
	learn_rate = 0.4
	layers = [
		NeuronLayer(2, 5),
		NeuronLayer(5, 2),
		NeuronLayer(2, 1),
	]
	layers = make_layers([2,5,2,1])
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
	plot_nn(nn, size, x, "line")


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
	size = 100
	x = logical_inputs(size)
	plot_nn(nn, size, x, "or")

def test_xor():
	nn = learn_xor()
	print("weights:\t", nn.get_weight())
	print("bias:   \t", nn.get_bias())
	size = 100
	x = logical_inputs(size)
	plot_nn(nn, size, x, "xor")

test_or()
test_line()
test_xor()
