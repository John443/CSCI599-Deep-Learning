import numpy as np
import copy


class sequential(object):
	def __init__(self, *args):
		"""
		Sequential Object to serialize the NN layers
		Please read this code block and understand how it works
		"""
		self.params = {}
		self.grads = {}
		self.layers = []
		self.paramName2Indices = {}
		self.layer_names = {}

		# process the parameters layer by layer
		layer_cnt = 0
		for layer in args:
			for n, v in layer.params.iteritems():
				if v is None:
					continue
				self.params[n] = v
				self.paramName2Indices[n] = layer_cnt
			for n, v in layer.grads.iteritems():
				self.grads[n] = v
			if layer.name in self.layer_names:
				raise ValueError("Existing name {}!".format(layer.name))
			self.layer_names[layer.name] = True
			self.layers.append(layer)
			layer_cnt += 1
		layer_cnt = 0

	def assign(self, name, val):
		# load the given values to the layer by name
		layer_cnt = self.paramName2Indices[name]
		self.layers[layer_cnt].params[name] = val

	def assign_grads(self, name, val):
		# load the given values to the layer by name
		layer_cnt = self.paramName2Indices[name]
		self.layers[layer_cnt].grads[name] = val

	def get_params(self, name):
		# return the parameters by name
		return self.params[name]

	def get_grads(self, name):
		# return the gradients by name
		return self.grads[name]

	def gather_params(self):
		"""
		Collect the parameters of every submodules
		"""
		for layer in self.layers:
			for n, v in layer.params.iteritems():
				self.params[n] = v

	def gather_grads(self):
		"""
		Collect the gradients of every submodules
		"""
		for layer in self.layers:
			for n, v in layer.grads.iteritems():
				self.grads[n] = v

	def load(self, pretrained):
		""" 
		Load a pretrained model by names 
		"""
		for layer in self.layers:
			if not hasattr(layer, "params"):
				continue
			for n, v in layer.params.iteritems():
				if n in pretrained.keys():
					layer.params[n] = pretrained[n].copy()
					print "Loading Params: {} Shape: {}".format(n, layer.params[n].shape)


class fc(object):
	def __init__(self, input_dim, output_dim, init_scale=0.02, name="fc"):
		"""
		In forward pass, please use self.params for the weights and biases for this layer
		In backward pass, store the computed gradients to self.grads
		- name: the name of current layer
		- input_dim: input dimension
		- output_dim: output dimension
		- meta: to store the forward pass activations for computing backpropagation
		"""
		self.name = name
		self.w_name = name + "_w"
		self.b_name = name + "_b"
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.params = {}
		self.grads = {}
		self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
		self.params[self.b_name] = np.zeros(output_dim)
		self.grads[self.w_name] = None
		self.grads[self.b_name] = None
		self.meta = None

	def forward(self, feat):
		""" Some comments """
		output = None
		assert np.prod(feat.shape[1:]) == self.input_dim, "But got {} and {}".format(
			np.prod(feat.shape[1:]), self.input_dim)
		#############################################################################
		# TODO: Implement the forward pass of a single fully connected layer.       #
		# You will probably need to reshape (flatten) the input features.           #
		# Store the results in the variable output provided above.                  #
		#############################################################################
		output = [np.matmul(f.reshape(self.input_dim), self.params[self.w_name]) + self.params[self.b_name] for f in feat]
		output = np.array(output)
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = feat
		return output

	def backward(self, dprev):
		""" Some comments """
		feat = self.meta
		if feat is None:
			raise ValueError("No forward function called before for this module!")
		dfeat, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
		#############################################################################
		# TODO: Implement the backward pass of a single fully connected layer.      #
		# You will probably need to reshape (flatten) the input gradients.          #
		# Store the computed gradients for current layer in self.grads with         #
		# corresponding name.                                                       # 
		#############################################################################
		dfeat = [np.matmul(self.params[self.w_name], item) for item in dprev]
		dfeat = np.array(dfeat)
		dfeat = np.reshape(dfeat, feat.shape)
		reshaped_feat = feat.reshape(feat.shape[0], np.prod(feat.shape[1:]))
		self.grads[self.w_name] = np.matmul(np.transpose(reshaped_feat), dprev)
		self.grads[self.b_name] = np.sum(dprev, axis=0)
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = None
		return dfeat


class relu(object):
	def __init__(self, name="relu"):
		"""
		- name: the name of current layer
		Note: params and grads should be just empty dicts here, do not update them
		"""
		self.name = name
		self.params = {}
		self.grads = {}
		self.grads[self.name] = None
		self.meta = None

	def forward(self, feat):
		""" Some comments """
		output = None
		#############################################################################
		# TODO: Implement the forward pass of a rectified linear unit               #
		# Store the results in the variable output provided above.                  #
		#############################################################################
		output = copy.deepcopy(feat)
		output = feat * (feat >= 0)
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = feat
		return output

	def backward(self, dprev):
		""" Some comments """
		feat = self.meta
		if feat is None:
			raise ValueError("No forward function called before for this module!")
		dfeat = None
		#############################################################################
		# TODO: Implement the backward pass of a rectified linear unit              #
		#############################################################################
		dfeat = copy.deepcopy(dprev)
		# row, col = np.shape(dfeat)[0], np.shape(dfeat)[1]
		# for i in range(row):
		# 	for j in range(col):
		# 		if feat[i][j] < 0:
		# 			dfeat[i][j] = 0
		dfeat = (feat >= 0) * dprev
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.meta = None
		return dfeat


class dropout(object):
	def __init__(self, p, seed=None, name="dropout"):
		"""
		- name: the name of current layer
		- p: the dropout probability
		- seed: numpy random seed
		- meta: to store the forward pass activations for computing backpropagation
		- dropped: the mask for dropping out the neurons
		- is_Training: dropout behaves differently during training and testing, use
		               this to indicate which phase is the current one
		"""
		self.name = name
		self.params = {}
		self.grads = {}
		self.grads[self.name] = None
		self.p = p
		self.seed = seed
		self.meta = None
		self.dropped = None
		self.is_Training = False

	def forward(self, feat, is_Training=True):
		if self.seed is not None:
			np.random.seed(self.seed)
		dropped = None
		output = None
		#############################################################################
		# TODO: Implement the forward pass of Dropout                               #
		#############################################################################
		retain_p = 1 - self.p
		# dropped = np.random.binomial(n=1, p=retain_p, size=feat.shape)
		# dropped = [np.random.binomial(n=1, p=retain_p, size=f.shape) for f in feat]
		# dropped = np.array(dropped)
		dropped = (np.random.rand(*feat.shape) < retain_p) / retain_p
		output = dropped * feat
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.dropped = dropped
		self.is_Training = is_Training
		self.meta = feat
		return output

	def backward(self, dprev):
		feat = self.meta
		dfeat = None
		if feat is None:
			raise ValueError("No forward function called before for this module!")
		#############################################################################
		# TODO: Implement the backward pass of Dropout                              #
		#############################################################################
		dropped = self.dropped
		# if self.p != 0:
		# 	dropped[dropped == 1] = 1 / self.p
		dfeat = dropped * dprev
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.is_Training = False
		return dfeat


class cross_entropy(object):
	def __init__(self, dim_average=True):
		"""
		- dim_average: if dividing by the input dimension or not
		- dLoss: intermediate variables to store the scores
		- label: Ground truth label for classification task
		"""
		self.dim_average = dim_average  # if average w.r.t. the total number of features
		self.dLoss = None
		self.label = None

	def forward(self, feat, label):
		""" Some comments """
		scores = softmax(feat)
		loss = None
		#############################################################################
		# TODO: Implement the forward pass of an CE Loss                            #
		#############################################################################
		reg = 1e-5
		(row, col) = feat.shape
		# one_hot_label = np.eye(col)[label]
		# for i in range(row):
		# 	loss.append(np.sum(np.nan_to_num(-one_hot_label[i] * np.log(scores[i] + h)), axis=0))
		# loss = np.array(loss)
		loss = -np.sum(np.log(scores[range(row), list(label)]))
		loss /= row
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.dLoss = scores.copy()
		self.label = label
		return loss

	def backward(self):
		dLoss = self.dLoss
		if dLoss is None:
			raise ValueError("No forward function called before for this module!")
		#############################################################################
		# TODO: Implement the backward pass of an CE Loss                           #
		#############################################################################
		(row, col) = dLoss.shape
		# one_hot_label = np.eye(col)[self.label]
		# dLoss = [dLoss[i] - one_hot_label[i] for i in range(row)]
		dLoss[range(row), list(self.label)] += -1
		dLoss = dLoss / row
		#############################################################################
		#                             END OF YOUR CODE                              #
		#############################################################################
		self.dLoss = dLoss
		return dLoss


def softmax(feat):
	""" Some comments """
	scores = None
	#############################################################################
	# TODO: Implement the forward pass of a softmax function                    #
	#############################################################################
	# normed_feat = []
	# for item in feat:
	# 	max_val = item.max()
	# 	min_val = item.min()
	# 	normed_feat.append((item - min_val) / (max_val - min_val))
	# normed_feat = np.array(normed_feat)
	# scores = [np.exp(item) / np.sum(np.exp(item)) for item in normed_feat]
	normed_feat = feat.copy()
	shift_feat = normed_feat - np.max(normed_feat, axis=1).reshape(-1, 1)
	scores = np.exp(shift_feat) / np.sum(np.exp(shift_feat), axis=1).reshape(-1, 1)
	# scores = [np.exp(item) / np.sum(np.exp(item)) for item in feat]
	# scores = np.array(scores)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return scores