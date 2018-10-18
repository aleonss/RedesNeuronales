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

		z = -summ -self.bias
		exp =0.0
		if(z<-700):
			exp = 0.0
		elif(z>700):
			exp = 1.0
		else:
			exp = math.exp(z)
		res = 1.0 / (1.0 + exp )
		return res

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

	def error_backpropagation(self,delta,output,nweight):
		ndelta=[]
		for i,n in enumerate(self.neurons):
			ne=0.0
			for j,w in enumerate(nweight):
				ne+= nweight[j][i]*delta[j]
			nd=ne*(output[i] * (1.0 - output[i]))			# o=np.multiply(output,(1.0 -output), dtype=np.float64)
			ndelta.append(nd)
		return ndelta

	def upgrade_wb(self, delta, input, rate):
		for i,n in enumerate(self.neurons):
			#for j,w in enumerate(n.weights):
			#	self.neurons[i].weights[j] += (rate * delta[i]*input[j])
			#self.neurons[i].bias += (rate*delta[i])
			self.neurons[i].upgrade_wb(delta[i], input, rate)
		return self.feed(input)


	def get_weight(self):
		res = []
		for n in self.neurons:
			res.append(n.weights)
		return res





class NeuralNetwork:

	def __init__(self,layers):
		self.layers=layers

	def forward_feeding(self,input):
		res = []
		outputs = []
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
			#print("eo ,ro: ",expected_output[i],", ",output[i])
			e=(expected_output[i] - output[i])
			d = e*(output[i] * (1.0 - output[i]))					#d = e* transferDerivative(output[i])
			#print("err ",e)
			#print("d ",d)

			error.append(e)
			delta.append(d)

		deltam.append(delta)

		for i in range(2,n_layers):
			l = self.layers[n_layers-i]
			nl = self.layers[n_layers-i +1]
			loutput = outputs[n_layers-i]
			nweight = nl.get_weight()
			delta = l.error_backpropagation(delta,loutput,nweight)
			deltam.append(delta)

		return deltam[::-1]

	def upgrade_wb(self, deltam, input, learn_rate):
		n_layers = self.layers.__len__()
		#print(input)

		for i,l in enumerate(self.layers[1:]):
			input=l.upgrade_wb(deltam[i], input, learn_rate)
			#print(input)









def realVal_2DLine(n_input):
	inputs = []
	results = []
	for i in range(0,n_input):
		xx= random.uniform(-50.0, 50.0)+0.0
		xy= random.uniform(-50.0, 50.0)

		inputs.append([xx,xy])
		results.append(0.0+xx<xy)

	return inputs,results


def plot_learned_2DLine(p,n_input,x,title):
	fig, ax = plt.subplots()
	color = ['red', 'blue']

	for j in range(0, n_input):
		rOutput = p.feed(x[j])
		c = rOutput
		xx = x[j][0]
		yy = x[j][1]
		scale = 20
		ax.scatter(xx, yy, c=color[c], s=scale, alpha=0.5, edgecolors='none')

	plt.title(title)
	plt.show()



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




def learn_2DLine():
	size = 1000
	trainings= 50
	lr=0.1

	sn = SigmoidNeuron()
	sn.mrand(2)


	for i in range(0,trainings):

		success=0.0
		x,dOutput = realVal_2DLine(size)
		print("x",x)

		#if (i % 10 == 0):
		#	plot_learned_2DLine(sn, size, x, i.__str__() + " trainings")


		for j in range(0, size):
			rOutput = sn.feed(x[j])
			#print("rOut",rOutput)
			diff =  dOutput[j] - rOutput

			for k in range(0, 2):
				sn.weights[k] += (lr*diff*x[j][k])

			if(diff==0):
				success+=1.0

		if (i % 10 == 0):
			ratio= success/(size+0.0)
			print(i,"\t",ratio)





#layer(size/n input, n neurons)

def learn_deep():
	size = 1000
	trainings= 500
	learn_rate = 0.5
	layers = [
		NeuronLayer(0, 2),
		NeuronLayer(2, 3),
		NeuronLayer(3, 4),
		NeuronLayer(4, 1),
	]
	nn = NeuralNetwork(layers)

	for i in range(0,trainings):

		success=0.0
		x,dOutput = realVal_2DLine(size)
		#print("x",x)
		if (i % 10 == 0):
			plot_deep_2DLine(nn, size, x, i.__str__() + " trainings")

		for j in range(0, size):
			res,outputs = nn.forward_feeding(x[j])
			print(dOutput[j],"\t",res[0],)

		for j in range(0, size):

			res,outputs = nn.forward_feeding(x[j])
			print(res)
			deltam=nn.error_backpropagation(outputs,dOutput)
			nn.upgrade_wb(deltam,x[j],learn_rate)

			diff = abs(dOutput[j]-res[0])
			success+= (1.0 - diff)

		if (i % 10 == 0):
			ratio= success/(size+0.0)
			print(i,"\t",ratio)

learn_deep()

'''
learn_2DLine()
learn_deep()


trainings = 50
learn_rate = 0.5
layers = [
	NeuronLayer(0, 5),
	NeuronLayer(5, 3),
	NeuronLayer(3, 4),
	NeuronLayer(4, 1),
]
nn = NeuralNetwork(layers)

input = [0.5, 1.0, 0.5, 0.5, 0.0]
expected_output = [0.5]

res, outputs = nn.forward_feeding(input)
deltam = nn.error_backpropagation(outputs, expected_output)
nn.upgrade_wb(deltam, input, learn_rate)



learn_2DLine()
if __name__ == '__main__':
	learn_2DLine()
	unittest.main()
'''
