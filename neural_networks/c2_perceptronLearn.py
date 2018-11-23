import unittest
import random


import matplotlib.pyplot as plt
from numpy.random import rand

class Perceptron:

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
		res =0.0
		for i,w in enumerate(self.weights):
			res+= w*x[i]
		return (res+self.bias) >= 0



class TestPerceptron(unittest.TestCase):

	def test_or(self):
		w = [1.0,1.0]
		b = -0.5
		p = Perceptron(w,b)

		for i in range(0,2):
			for j in range(0,2):
				x=[i,j]
				real = i or j
				result = p.feed(x)
				self.assertEqual(real, result)	#print(x,"\t",result)


	def test_and(self):
		w = [1.0,1.0]
		b = -1.5
		p = Perceptron(w,b)

		for i in range(0,2):
			for j in range(0,2):
				x=[i,j]
				real = i and j
				result = p.feed(x)
				self.assertEqual(real, result)

	def test_not(self):
		w = [-1.0]
		b = 0.5
		p = Perceptron(w,b)

		for i in range(0,2):
			x=[i]
			real = not i
			result = p.feed(x)
			self.assertEqual(real, result)

	def test_nand(self):
		w = [-2.0,-2.0]
		b = 3
		p = Perceptron(w,b)

		for i in range(0,2):
			for j in range(0,2):
				x=[i,j]
				real = not(i and j)
				result = p.feed(x)
				self.assertEqual(real, result)

	def test_sum(self):
		for i in range(0,2):
			for j in range(0,2):

				real_summ = (i+j)%2
				real_carr = (i+j)>1

				res_summ, res_carr = sum_nand(i,j)

				self.assertEqual(real_summ, res_summ)
				self.assertEqual(real_carr, res_carr)



def sum_nand(x1,x2):
	w = [-2.0,-2.0]
	b = 3
	p = Perceptron(w,b)

	r0 = p.feed([x1,x2])
	r1 = p.feed([x1,r0])
	r2 = p.feed([r0,x2])

	summ = p.feed([r1,r2]) +0
	carr = p.feed([r0,r0]) +0

	return summ, carr



def realVal_2DLine(size):
	inputs = []
	results = []
	for i in range(0,size):
		xx= random.uniform(-50.0, 50.0)+0.0
		xy= random.uniform(-50.0, 50.0)

		inputs.append([xx,xy])
		results.append(0.0+xx<xy)

	return inputs,results


def plot_learned_2DLine(p,size,x,title):
	fig, ax = plt.subplots()
	color = ['red', 'blue']

	for j in range(0, size):
		rOutput = p.feed(x[j])
		c = rOutput
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

	p = Perceptron()
	p.mrand(2)


	for i in range(0,trainings):

		success=0.0
		x,dOutput = realVal_2DLine(size)

		if (i % 10 == 0):
			plot_learned_2DLine(p, size, x, i.__str__() + " trainings")


		for j in range(0, size):
			rOutput = p.feed(x[j])
			diff =  dOutput[j] - rOutput

			for k in range(0, 2):
				p.weights[k] += (lr*diff*x[j][k])

			if(diff==0):
				success+=1.0

		if (i % 10 == 0):
			ratio= success/(size+0.0)
			print(i,"\t",ratio)
	return p




p = learn_2DLine()

'''
if __name__ == '__main__':
	learn_2DLine()
	unittest.main()
'''


