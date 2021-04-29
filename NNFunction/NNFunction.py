from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import KFold
import copy
import PyFileIO as pf
import os
import matplotlib.pyplot as plt 

class NNFunction(object):
	def __init__(self,s,AF='softplus',Output='linear',Loss='mean_squared_error'):
		'''
		This object is used to train a neural network to reproduce 
		non-linear functions.
		
		Inputs
		======
		s : int
			Integer array defining the number of nodes in each layer,
			where s[0] is the input layer, s[-1] is the output layer,
			and s[1:-1] are the hidden layers.
		AF : str
			This defines the activation functions which will be used for
			the hidden layers. Can be:
			'softplus'|'relu'|'LeakyReLU'|'sigmoid'|'softmax'|'softplus'
			'softsign'|'tanh'|'selu'|'elu'|'linear'
		Output : str
			Output layer activation functions - same as those for AF.
		Loss : str
			The loss function
			'mean_squared_error'|'mean_absolute_error' - see Keras
			documentation for more.

			
		
		'''
		#store some defining parameters
		self.L = np.size(s)
		self.s = s
		self.Loss = Loss

		#define the activation functions
		if AF == 'LeakyReLU':
			self.HidAF = layers.LeakyReLU()
		else:
			self.HidAF = AF
		if Output == 'LeakyReLU':
			self.OutAF = layers.LeakyReLU()
		else:
			self.OutAF = Output
		
		

		#initialize the model
		self.model = None
		self.k = 0

		self.val = None
		self.test = None
		
	def __del__(self):
		del self.model
		
	def _DefineLayers(self):
		'''
		Define the layers to be used for this ANN
		
		'''
		inputs = keras.Input(shape=(self.s[0],))
		prev = inputs
		for i in range(1,self.L-1):
			x = layers.Dense(self.s[i],activation=self.HidAF)(prev)
			prev = x
		outputs = layers.Dense(self.s[-1],activation=self.OutAF)(prev)	
		return inputs,outputs
		
	def _CreateModel(self):
		'''
		This function will create an neural network model based upon the
		parameters stored in the object.
		
		'''
		inputs, outputs = self._DefineLayers()
		model = keras.Model(inputs=inputs,outputs=outputs)
		model.compile(optimizer='Adam',loss=self.Loss,metrics=[self.Loss])			
		return model
	
	def CreateData(self,Func,Range=[-5.0,5.0],RescaleY=True,n=100):
		'''
		This function is mostly used for testing - it can be used to 
		generate training data given a callable function.
		
		Inputs
		======
		Func : callable
			Function which accepts a 2D array (matrix) of input features
			which would be used to train the network.
		Range : list
			2-element list which defines the range of X input values.
		RescaleY : bool
			Rescale the outputs of Func to be within the range -1 < y < 1
		n : int
			The number of samples to create.
		
		
		'''
		
		X = np.linspace(Range[0],Range[1],n)
		y = Func(X)
		
		X = np.array([X]).T
		nd = np.size(y.shape)
		if nd == 1:
			y = np.array([y])
		ny = y.shape[0]
		
		for i in range(0,ny):
			yrnge = [y[i].min(),y[i].max()]
			noise = np.random.randn(y[i].size)*(yrnge[1] - yrnge[0])/10.0
			y[i] += noise
		
		self.AddData(X,y,RescaleY)	

	def CreateTestData(self,Func,Range=[-5.0,5.0],RescaleY=True,n=100):
		'''
		This function is mostly used for testing - it does the same as
		CreateData, but this creates data to be used as a test set.
		
		Inputs
		======
		Func : callable
			Function which accepts a 2D array (matrix) of input features
			which would be used to train the network.
		Range : list
			2-element list which defines the range of X input values.
		RescaleY : bool
			Rescale the outputs of Func to be within the range -1 < y < 1
		n : int
			The number of samples to create.
		
		
		'''

		X = np.linspace(Range[0],Range[1],n)
		y = Func(X)
		
		X = np.array([X]).T
		nd = np.size(y.shape)
		if nd == 1:
			y = np.array([y])
		ny = y.shape[0]
		
		for i in range(0,ny):
			yrnge = [y[i].min(),y[i].max()]
			noise = np.random.randn(y[i].size)*(yrnge[1] - yrnge[0])/10.0
			y[i] += noise
		
		self.AddTestData(X,y,RescaleY)		
		
		
	def AddData(self,X,y,RescaleY=True,SampleWeights=None):
		'''
		Add training data matrices to the object.
		
		NOTE: When using kfolds > 1 during training, this will be split
		into training and validation sets.
		
		Inputs
		======
		X : float
			Input matrix, shape (m,n), where m is the number of samples
			and n is the number of input nodes (==s[0])
		y : float
			Hypothesis matrix, shape (m,k), m is the number of samples
			and k is the number of output nodes (==s[-1])
		RescaleY : bool
			Rescale the outputs of Func to be within the range -1 < y < 1
		SampleWeights : None|float
			If None - all samples are weighted equally, otherwise set to
			an array with shape (m,), defining the weight of each 
			individual sample.

		'''
		if np.size(X.shape) == 1:
			self.X = np.array([X]).T
		else:
			self.X = np.array(X)
		
		if np.size(y.shape) == 1:
			self.y = np.array([y]).T
		else:
			self.y = np.array(y)
			
		if RescaleY:
			mx = self.y.max(axis=0)
			mn = self.y.min(axis=0)
			self.scale0 = 2.0/(mx - mn)
			self.scale1 = 0.5*(mx + mn)
			self.y = (self.y - self.scale1) * self.scale0
		else:
			self.scale0 = 1.0
			self.scale1 = 0.0
		
		self.SampleWeights = SampleWeights
	
	def AddValidationData(self,X,y,RescaleY=True):
		'''
		Add validaton data matrices to the object.
		
		NOTE - if using kfolds > 1 when training, these data will not be
		used. The Validation will come from the training matrices 
		instead.
		
		Inputs
		======
		X : float
			Input matrix, shape (m,n), where m is the number of samples
			and n is the number of input nodes (==s[0])
		y : float
			Hypothesis matrix, shape (m,k), m is the number of samples
			and k is the number of output nodes (==s[-1])
		RescaleY : bool
			Rescale the outputs of Func to be within the range -1 < y < 1

		'''
		if np.size(X.shape) == 1:
			self.Xcv = np.array([X]).T
		else:
			self.Xcv = np.array(X)
		
		if np.size(y.shape) == 1:
			self.ycv = np.array([y]).T
		else:
			self.ycv = np.array(y)
			

		self.ycv = (self.ycv - self.scale1) * self.scale0
		self.val = (self.Xcv,self.ycv)
		
	def AddTestData(self,X,y,RescaleY=True):
		'''
		Add test data matrices to the object.
		
		NOTE - These data are ONLY used for testing, not for training.
		
		Inputs
		======
		X : float
			Input matrix, shape (m,n), where m is the number of samples
			and n is the number of input nodes (==s[0])
		y : float
			Hypothesis matrix, shape (m,k), m is the number of samples
			and k is the number of output nodes (==s[-1])
		RescaleY : bool
			Rescale the outputs of Func to be within the range -1 < y < 1

		'''
		if np.size(X.shape) == 1:
			self.Xts = np.array([X]).T
		else:
			self.Xts = np.array(X)
		
		if np.size(y.shape) == 1:
			self.yts = np.array([y]).T
		else:
			self.yts = np.array(y)
			

		self.yts = (self.yts - self.scale1) * self.scale0
		self.test = (self.Xts,self.yts)

	def _InitNetwork(self,kfolds):
		'''
		Initialize the network(s).
		
		Inputs
		======
		kfolds : int
			Number of kfolds to do.
			By default is it 1 - this will use separate training and 
			validation data sets where available; only one self.Model is 
			created.
			When kfolds > 1 - this will only use data added using the
			AddData method, where the data will be split into kfolds 
			equal sets. Each of these sets will take turns being the 
			validation set, and results in kfolds networks in self.Model
				
		'''
		#initialize model upon first train, in case we decide to do
		#k-folds or not
		if self.model is None:
			if not self.val is None or kfolds == 1:
				self.k = 1
				self.model = [self._CreateModel()]
				self.Jt = [np.array([],dtype='float32')]
				self.Jc = [np.array([],dtype='float32')]
				self.hist = []
				if kfolds != 1:
					print('Validation data defined, using single k-fold')
			else:
				self.k = kfolds
				self.model = []
				for i in range(0,self.k):
					self.model.append(self._CreateModel())
				self.Jt = [np.array([],dtype='float32')]*self.k
				self.Jc = [np.array([],dtype='float32')]*self.k
				self.hist = np.zeros(self.k,dtype='object')
				for i in range(0,self.k):
					self.hist[i] = []			

	def Train(self,nEpoch,BatchSize=None,verbose=1,kfolds=1):
		'''
		Train the network using the provided datasets.
		
		Inputs
		======
		nEpoch : int
			The total number of epochs.
		BatchSize : None|int
			The size of each mini-batch to be used during training. 
		verbose : int
			As in keras: 0 = silent; 1 = progress bar; 2 = a line per
			epoch.
		kfolds : int
			Number of kfolds to do.
			By default is it 1 - this will use separate training and 
			validation data sets where available; only one self.Model is 
			created.
			When kfolds > 1 - this will only use data added using the
			AddData method, where the data will be split into kfolds 
			equal sets. Each of these sets will take turns being the 
			validation set, and results in kfolds networks in self.Model
		
		'''
		#intialize the networks
		self._InitNetwork(kfolds)

		#now we can train the network(s)
		if self.k == 1:
			hist = self.model[0].fit(self.X,self.y,epochs=nEpoch,batch_size=BatchSize,validation_data=self.val,verbose=verbose,sample_weight=self.SampleWeights)
			self.Jt[0] = np.append(self.Jt[0],hist.history['loss'])
			if not self.val is None:
				self.Jc[0] = np.append(self.Jc[0],hist.history['val_loss'])
			self.hist.append(hist)
		else:
			kf = KFold(n_splits=self.k)
			k = 0
			for train_index, test_index in kf.split(self.X):
				print('K-fold {:d} of {:d}'.format(k+1,self.k))
				Xt = self.X[train_index]
				Xc = self.X[test_index]
				yt = self.y[train_index]
				yc = self.y[test_index]
				
				if self.SampleWeights is None:
					sw = None
				else:
					print('Using sample weights')
					sw = copy.deepcopy(self.SampleWeights[train_index])
				
				hist = self.model[k].fit(Xt,yt,epochs=nEpoch,batch_size=BatchSize,validation_data=(Xc,yc),verbose=verbose,sample_weight=sw)

				self.Jt[k] = np.append(self.Jt[k],hist.history['loss'])
				self.Jc[k] = np.append(self.Jc[k],hist.history['val_loss'])
				self.hist[k].append(hist)
				k+=1
		return self.hist
		
	def Predict(self,X,RescaleY=True,k=0):
		'''
		Predict the output of the neural network given an input array.
		
		Inputs
		======
		X : float
			Input matrix, shape (m,n), where m is the number of samples
			and n is the number of input nodes (==s[0])
		RescaleY : bool
			Rescale the outputs of Func to be within the range -1 < y < 1	
		
		Returns
		=======
		y : float
			Output array from the network, shape (m,k) where m is the 
			number of samples and k is the number of output nodes.
		
		'''
		if np.size(X.shape) == 1:
			x = np.array([X]).T
		else:
			x = X
		y = self.model[k].predict(x)
		if RescaleY:
			y = y/self.scale0 + self.scale1
		return y
	
	def Test(self,X=None,y=None,k=0):
		'''
		Test the ANN on a set of test inputs and outputs.
		
		Inputs
		======
		X : float
			Input matrix, shape (m,n), where m is the number of samples
			and n is the number of input nodes (==s[0])
		y : float
			Hypothesis matrix, shape (m,k), m is the number of samples
			and k is the number of output nodes (==s[-1])
		k : int
			The network to test.
			
		Returns
		=======
		out : float
			Cost function output.
		'''
		if X is None and not self.test is None:
			Xt = self.test[0]
		else:
			Xt = X
			
		if y is None and not self.test is None:
			yt = self.test[1]
		else:
			yt = y
			
		if Xt is None or yt is None:
			print('Please add test data')
			return None
		
		out = self.model[k].evaluate(x=Xt,y=yt,verbose=0)
		return out[0]
			
	
	def GetWeights(self,k=0):
		'''
		Retreive the weight and bias matrics from a network.
		
		Inputs
		======
		k : int
			Network index.
			
		Returns
		=======
		w : list
			list of 2D wight matrices
		b : list
			list of bias matrices
		
		'''
		w = []
		b = []
		tmp = self.model[k].get_weights()
		for i in range(0,self.L-1):
			w.append(tmp[i*2])
			b.append(tmp[i*2+1])
		return w,b
		
	def SetWeights(self,w,b,k=0):
		'''
		Set the weight and bias matrics for a network.
		
		Inputs
		======
		w : list
			list of 2D wight matrices
		b : list
			list of bias matrices
		k : int
			Network index.
	
		'''

		if self.model is None:
			self.model = [self._CreateModel()]

		ipt = []
		for i in range(0,self.L-1):
			ipt.append(w[i])
			ipt.append(b[i])
		self.model[k].set_weights(ipt)
		
	def Save(self,fname=None,Best=False,ferr=None):
		'''
		Saves artificial neural network(s) to file(s)
		
		Inputs
		======
		fname : str
			Full path and name of file to be saved
		Best : bool
			If True, the best of the networks will be the only one saved
		ferr : None|str
			Name of the file to store test scores in.
		
	
		'''
		#get the file name
		if fname is None:
			fname = ''
			for s in self.s:
				fname += '_{:d}'.format(s)
			fname = fname[1:]
		else:
			p = fname.rfind('/')
			path = fname[:p]
			if not os.path.isdir(path) and path != '':
				os.system('mkdir -pv '+path)
		
		if not fname[-4:] == '.bin':
			 fname += '.bin'
		fout = fname
		
		#choose whether to save all anns or just the best
		if self.k == 1 and not Best:
			#only one to save
			self._SaveANN(0,fout)
		elif Best:
			J = [0]*self.k
			#choose the best performing ann
			if not self.test is None:
				for k in range(0,self.k):
					J[k] = self.Test(k=k)
			else:
				for k in range(0,self.k):
					J[k] = self.Jc[k][-1]
			J = np.array(J)
			use = np.argmin(J)
			
			#get previous tests
			if ferr is None:
				ferr = fname[:-4]+'.err'
			errors = self._GetTests(ferr)
			
			#check if the newest error is less then the rest
			if errors.size == 0:
				savenet = True
			elif J[use] < np.nanmin(errors):
				savenet = True
			else:
				savenet = False			
			
			#save tests
			self.SaveTests(ferr)
			
			if savenet:
				self._SaveANN(use,fout)
		else:
			#save all
			for k in range(0,self.k):
				self._SaveANN(k,fout+'.{:01d}'.format(k))
				
	def SaveTests(self,fname=None):
		'''
		Save the test scores to file.
		
		Inputs
		======
		fname : None|str
			Name of output file.
		
		'''
		#get the file name
		if fname is None:
			fname = ''
			for s in self.s:
				fname += '_{:d}'.format(s)
			fname = fname[1:]
		else:
			p = fname.rfind('/')
			path = fname[:p]
			if not os.path.isdir(path) and path != '':
				os.system('mkdir -pv '+path)
		if not fname[-4:] == '.err':
			fname += '.err'
		fout = fname		
		
		self._SaveTest(fout)
	
	
	def _GetTests(self,fname):
		'''
		Read the test scores from file.
		
		Inputs
		======
		fname : str
			File name and path
		
		Returns
		=======
		err : float
			numpy.ndarray of test scores
		
		'''
		try:
			f = open(fname,'rb')
			errors = np.fromfile(f,dtype='float32')
			f.close()
		except:
			errors = np.array([]).astype('float32')		
		return errors
				
	def _SaveTest(self,fname):
		'''
		Save the test scores to file.
		
		Inputs
		======
		fname : str
			File name and path		
		
		'''
		err = [0]*self.k

		if not self.test is None:
			for k in range(0,self.k):
				err[k] = self.Test(k=k)
		else:
			for k in range(0,self.k):
				err[k] = self.Jc[k][-1]

		err = np.array(err)
		
		errors = self._GetTests(fname)
		errors = np.append(errors,err)
			
		#re-save the errors
		print('Saving errors {:s}'.format(fname))
		f = open(fname,'wb')
		errors.astype('float32').tofile(f)
		f.close()				
		
				
	def _SaveANN(self,k,fname):
		'''
		Saves artificial neural network to file
		
		Inputs
		======
		k : int
			Network array index
		fname : str
			Full path and name of file to be saved		
		
		'''
		print('Saving {:s}'.format(fname))
		#save the network
		#return the weights
		w,b = self.GetWeights(k)
				
		#now save the result
		f = open(fname,'wb')
		pf.ArrayToFile(self.s,'int32',f)
		pf.ArrayToFile(self.scale0,'float32',f)
		pf.ArrayToFile(self.scale1,'float32',f)
		pf.ListArrayToFile(w,'float32',f)
		pf.ListArrayToFile(b,'float32',f)
		pf.ArrayToFile(self.Jt[k],'float32',f)
		pf.ArrayToFile(self.Jc[k],'float32',f)
		f.close()
		

	def PlotCost(self,k=0,fig=None,maps=[1,1,0,0]):
		'''
		Plot the cost function for a trained network.
		
		Inputs
		======
		k : int
			Network index
		fig : obj|None
			If None, a new figure will be created; if it is an instance
			of matplotlib.pyplot then a new subplot will be created on 
			the current figure; if it is an instance of pylot.Axes, then
			the current axes will be used.
		maps:
			Grid location on the plot: [xmaps,ymaps,xmap,ymap]
		
		'''
		if fig is None:
			fig = plt
			fig.figure()
		if hasattr(fig,'Axes'):	
			ax = fig.subplot2grid((maps[1],maps[0]),(maps[3],maps[2]))
		else:
			ax = fig		 
			
		ax.plot(self.Jt[k],color='blue',label='$J_t$')
		ax.plot(self.Jc[k],color='red',label='$J_c$')
		
		ax.set_xlim(0,self.Jt[k].size)
		yl = ax.get_ylim()
		ax.set_ylim(0,yl[-1])
		
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Cost')
		
		ax.legend()
				
		return ax
		
