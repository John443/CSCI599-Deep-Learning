import numpy as np


""" Super Class """
class Optimizer(object):
	""" 
	This is a template for implementing the classes of optimizers
	"""
	def __init__(self, net, lr=1e-4):
		self.net = net  # the model
		self.lr = lr    # learning rate

	""" Make a step and update all parameters """
	def step(self):
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				pass


""" Classes """
class SGD(Optimizer):
	""" Some comments """
	def __init__(self, net, lr=1e-4):
		self.net = net
		self.lr = lr

	def step(self):
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dv = layer.grads[n]
				layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
	def __init__(self, net, lr=1e-4, momentum=0.0):
		self.net = net
		self.lr = lr
		self.momentum = momentum
		self.velocity = {}

	def step(self):
		#############################################################################
		# TODO: Implement the SGD + Momentum                                        #
		#############################################################################
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dw = layer.grads[n]
				w = layer.params[n]
				velocity = self.velocity.get(n, np.zeros_like(w))
				self.velocity[n] = self.momentum * velocity - self.lr * dw
				layer.params[n] += self.velocity[n]
		# pass
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################


class RMSProp(Optimizer):
	def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
		self.net = net
		self.lr = lr
		self.decay = decay
		self.eps = eps
		self.cache = {}  # decaying average of past squared gradients

	def step(self):
		#############################################################################
		# TODO: Implement the RMSProp                                               #
		#############################################################################
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dw = layer.grads[n]
				w = layer.params[n]
				cache = self.cache.get(n, np.zeros_like(w))
				self.cache[n] = self.decay * cache + (1 - self.decay) * (dw ** 2)
				layer.params[n] = w - self.lr * dw / (np.sqrt(self.cache[n] + self.eps))
		# pass
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################


class Adam(Optimizer):
	def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
		self.net = net
		self.lr = lr
		self.beta1, self.beta2 = beta1, beta2
		self.eps = eps
		self.mt = {}
		self.vt = {}
		self.t = t

	def step(self):
		#############################################################################
		# TODO: Implement the Adam                                                  #
		#############################################################################
		for layer in self.net.layers:
			for n, v in layer.params.iteritems():
				dw = layer.grads[n]
				w = layer.params[n]
				mt = self.mt.get(n, np.zeros_like(w))
				vt = self.vt.get(n, np.zeros_like(w))
				self.t += 1
				self.mt[n] = self.beta1 * mt + (1 - self.beta1) * dw
				self.vt[n] = self.beta2 * vt + (1 - self.beta2) * (dw ** 2)
				mb = self.mt[n] / (1 - self.beta1 ** self.t)
				vb = self.vt[n] / (1 - self.beta2 ** self.t)
				layer.params[n] = w - self.lr * mb / (np.sqrt(vb) + self.eps)
		# pass
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################