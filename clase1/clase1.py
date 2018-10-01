import unittest


class Perceptron:
	
	def __init__(self,w,b):
		self.weights=w
		self.bias=b

	def feed(self,x):
		res =0
		for i,w in enumerate(self.weights):
			res+= w* x[i]
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
	




if __name__ == '__main__':
	unittest.main()

