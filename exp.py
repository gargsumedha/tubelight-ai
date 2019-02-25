from numpy import exp, array, random, dot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NeuralNetwork():
	def __init__(self):
		random.seed(1)

		# setting the number of nodes in layer 2 and layer 3
		# more nodes --> more confidence in predictions (?)
		l2 = 3
		l3 = 2

		# assign random weights to matrices in network
		# format is (no. of nodes in previous layer) x (no. of nodes in following layer)
		self.synaptic_weights1 = 2 * random.random((3, l2)) -1
		self.synaptic_weights2 = 2 * random.random((l2, l3)) -1
		self.synaptic_weights3 = 2 * random.random((l3, 1)) -1
		
	def __sigmoid(self, x):
		return 1/(1+exp(-x))

	# derivative of sigmoid function, indicates confidence about existing weight
	def __sigmoid_derivative(self, x):
		return x*(1-x)

	# train neural network, adusting synaptic weights each time
	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):

			# pass training set through our neural network
			# a2 means the activations fed to second layer
			a2 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
			a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
			output = self.__sigmoid(dot(a3, self.synaptic_weights3))

			# calculate 'error'
			del4 = (training_set_outputs - output)*self.__sigmoid_derivative(output)

			# find 'errors' in each layer
			del3 = dot(self.synaptic_weights3, del4.T)*(self.__sigmoid_derivative(a3).T)
			del2 = dot(self.synaptic_weights2, del3)*(self.__sigmoid_derivative(a2).T)

			# get adjustments (gradients) for each layer
			adjustment3 = dot(a3.T, del4)
			adjustment2 = dot(a2.T, del3.T)
			adjustment1 = dot(training_set_inputs.T, del2.T)

			# adjust weights accordingly
			self.synaptic_weights1 += adjustment1
			self.synaptic_weights2 += adjustment2
			self.synaptic_weights3 += adjustment3

	def forward_pass(self, inputs):
		# pass our inputs through our neural network
		a2 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
		a3 = self.__sigmoid(dot(a2, self.synaptic_weights2))
		output = self.__sigmoid(dot(a3, self.synaptic_weights3)) 
		return output

if __name__ == "__main__":
    # initialise single neuron neural network
    neural_network = NeuralNetwork()
    print("Random starting synaptic weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("\nRandom starting synaptic weights (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("\nRandom starting synaptic weights (layer 3): ")
    print(neural_network.synaptic_weights3)
    DataSet1 = [[0] * 3 for _ in range(100)]
    DataSet2 = [[0] * 3 for _ in range(100)]
    for i in range(100): 
        DataSet1[i][0] = round(random.uniform(2,3),2)
        DataSet1[i][1] = round(random.uniform(2,3),2)
        DataSet1[i][2] = round(random.random(),3)
    for i in range(100):
        DataSet2[i][0] = round(random.uniform(0.5,1.5),2)
        DataSet2[i][1] = round(random.uniform(0.5,1.5),2)
        DataSet2[i][2] = round(random.random(),2)

  
    # the training set.

    for l in range(100):
        training_set_inputs = array([ [ DataSet1[l][0],DataSet1[l][1],DataSet1[l][2] ], [ DataSet2[l][0],DataSet2[l][1],DataSet2[l][2] ] ])
        training_set_outputs = array([[1,0]]).T
        neural_network.train(training_set_inputs, training_set_outputs, 10000)


    print ("\nNew synaptic weights (layer 1) after training: ")
    print (neural_network.synaptic_weights1)
    print ("\nNew synaptic weights (layer 2) after training: ")
    print (neural_network.synaptic_weights2)
    print ("\nNew synaptic weights (layer 3) after training: ")
    print (neural_network.synaptic_weights3)
    # test with new input
    ch = 'y'
    while ch == 'y':
        print ("\nConsidering new situation -> ?")
        x1 =float( input("enter x coordinate"))
        x2 =float( input("enter y coordinate"))
        x3 =float(input("enter z coordinate"))
        op= float(neural_network.forward_pass(array([x1,x2,x3])))
        off = round(op,1)
        print(off)
        ch = input("press y to continue")
