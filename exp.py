from numpy import exp, array, random, dot, all
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
import numpy
from numpy.random import randn
from scipy import array, newaxis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
lr=0.001
class CreateNet():
	def __init__(self):
		random.seed(1)

		# The number of nodes in hidden layer 2 and hidden layer 3 are defined below
		hidden_layer2 = 3
		hidden_layer3 = 2
		
		# assigning random weights to 2D array in the network
		self.weights1 = 2 * random.random((3, hidden_layer2)) -1
		self.weights2 = 2 * random.random((hidden_layer2, hidden_layer3)) -1
		self.weights3 = 2 * random.random((hidden_layer3, 1)) -1
		
	def __sigmoid(self, x):
		return 1/(1+exp(-x))

	
	def __sigmoid_derivative(self, x):
		return x*(1-x)

	# training the network and adusting weights each time
	def training(self, inputs, outputs, iterations):
		print("Hello")
		for i in range(iterations):

			# passing the randomly generated training set through our neural network
			# activation2 is the activations fed to second layer
			activation2 = self.__sigmoid(dot(inputs, self.weights1))
			activation3 = self.__sigmoid(dot(activation2, self.weights2))
			output = self.__sigmoid(dot(activation3, self.weights3))
			#print(output)
			# error calculation
			del4 = (outputs - output)*self.__sigmoid_derivative(output)

			# errors in each layer
			del3 = dot(self.weights3, del4.T)*(self.__sigmoid_derivative(activation3).T)
			del2 = dot(self.weights2, del3)*(self.__sigmoid_derivative(activation2).T)

			# if we reach the required solution no need to feed forword
			
			# getting gradients for each layer
			adj3 = dot(activation3.T, del4)
			adj2 = dot(activation2.T, del3.T)
			adj1 = dot(inputs.T, del2.T)

			# adjusting weights accordingly
			self.weights1 += adj1
			self.weights2 += adj2
			self.weights3 += adj3

			
	def forward(self, inputs):
		# passing the inputs through the neural network
		a2 = self.__sigmoid(dot(inputs, self.weights1))
		a3 = self.__sigmoid(dot(a2, self.weights2))
		output = self.__sigmoid(dot(a3, self.weights3)) 
		return output

if __name__ == "__main__":
	output_s = [1]
	DataSets = [[round(random.uniform(2,3),2),round(random.uniform(2,3),2),round(random.random(),3)]]
	for i in range(1,100):
		output_s.extend([1]) 
		DataSets.insert(i,[round(random.uniform(2,3),2),round(random.uniform(2,3),2),round(random.random(),3)])
	output_s.extend([0])
	DataSets.insert(101,[round(random.uniform(0.5,1.5),2),round(random.uniform(0.5,1.5),2),round(random.random(),3)])
	for i in range(102,200):
		output_s.extend([0])
		DataSets.insert(i,[round(random.uniform(0.5,1.5),2),round(random.uniform(0.5,1.5),2),round(random.random(),3)])

	# initialise single neuron neural network
	for i in range(100):
	    xs, ys, zs = DataSets[i]
	    ax.scatter(xs, ys, zs, c="red", marker="^")
	for i in range(101,199):
	    xs, ys, zs = DataSets[i]
	    ax.scatter(xs, ys, zs, c="green", marker="^")

	
	neural_net = CreateNet()
	print("Starting weights for layer 1, layer 2 and layer 3: ")
	print(neural_net.weights1)
	print("\n")
	print(neural_net.weights2)
	print("\n")
	print(neural_net.weights3)
	
	inputs = array(DataSets)
	outputs = array([output_s]).T
	#print(inputs)
	#print(outputs)
	neural_net.training(inputs, outputs, 50000)
	print ("\nNew weights for layer 1, layer 2 and layer 3 after training: ")
	print (neural_net.weights1)
	print ("\n")
	print (neural_net.weights2)
	print ("\n")
	print (neural_net.weights3)
    # testing with new input


	ch = 'y'
	
	while ch == 'y':
		print ("\nCoordinates for new situation:")
		x1 =float( input("enter x coordinate"))
		x2 =float( input("enter y coordinate"))
		x3 =float(input("enter z coordinate"))
		ax.scatter(x1, x2, x3, c="blue", marker="^")
		op= float(neural_net.forward(array([x1,x2,x3])))
		print(op)
		#off = round(op,1)
		#if off <= 0.5:
		#	print(0)
		#else:
		#	print (1)
		#print(off)
		ch = raw_input("press y to continue")
	#DATA = array([ [0.08216108,  1.67946197, -4.02173908],[-1.82328547, -3.83638578,  0.7452409 ],[-2.7231652,  -0.56974737,  4.21862478]])
	DATA = array([[neural_net.weights1[0][0],neural_net.weights1[0][1],neural_net.weights1[0][2]],
	[neural_net.weights1[1][0],neural_net.weights1[1][1],neural_net.weights1[1][2]],
	[neural_net.weights1[2][0],neural_net.weights1[2][1],neural_net.weights1[2][2]]
	])
	Xs = DATA[:,0]
	Ys = DATA[:,1]
	Zs = DATA[:,2]
	surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
	fig.colorbar(surf)

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(6))
	ax.zaxis.set_major_locator(MaxNLocator(5))

	fig.tight_layout()	
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()
