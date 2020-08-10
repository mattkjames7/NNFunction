from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import KFold
import copy

class NNFunction(object):
	def __init__(self,s,AF='softplus',Output='linear',Loss='mean_squared_error'):
	
		self.L = np.size(s)
		self.s = s

		#define the activation functions
		if AF == 'LeakyReLU':
			HidAF = layers.LeakyReLU()
		else:
			HidAF = AF
		
		#now try and create a model
		inputs = keras.Input(shape=(s[0],))
		prev = inputs
		for i in range(1,self.L-1):
			x = layers.Dense(s[i],activation=HidAF)(prev)
			prev = x
		outputs = layers.Dense(s[-1],activation=Output)(prev)
		self.model = keras.Model(inputs=inputs,outputs=outputs)
		self.model.compile(optimizer='Adam',loss=Loss,metrics=[Loss])	
		self.val = None
		self.Jt = np.array([],dtype='float32')
		self.Jc = np.array([],dtype='float32')
		self.hist = []
		
	def __del__(self):
		del self.model
	
	def CreateData(self,Func,Range=[-5.0,5.0],RescaleY=True):
		X = np.linspace(Range[0],Range[1],100)
		y = Func(X)
		
		X = np.array([X]).T
		nd = np.size(y.shape)
		if nd == 1:
			y = np.array([y])
		ny = y.shape[0]
		
		print(X.shape,y.shape)
		for i in range(0,ny):
			yrnge = [y[i].min(),y[i].max()]
			noise = np.random.randn(y[i].size)*(yrnge[1] - yrnge[0])/10.0
			y[i] += noise
		
		self.AddData(X,y,RescaleY)	
		
		
	def AddData(self,X,y,RescaleY=True,SampleWeights=None):
		'''
		Input shape (m,n)
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
		Input shape (m,n)
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

		
	def Train(self,nEpoch,BatchSize=None,verbose=1,kfolds=1):
		
		if kfolds == 1:
			hist = self.model.fit(self.X,self.y,epochs=nEpoch,batch_size=BatchSize,validation_data=self.val,verbose=verbose,sample_weight=self.SampleWeights)
			self.Jt = np.append(self.Jt,hist.history['loss'])
			self.Jc = np.append(self.Jc,hist.history['val_loss'])
			self.hist.append(hist)
		else:
			kf = KFold(n_splits=kfolds)
			k = 0
			for train_index, test_index in kf.split(self.X):
				print('K-fold {:d} of {:d}'.format(k+1,kfolds))
				Xt = self.X[train_index]
				Xc = self.X[test_index]
				yt = self.y[train_index]
				yc = self.y[test_index]
				
				if self.SampleWeights is None:
					sw = None
				else:
					print('Using sample weights')
					sw = copy.deepcopy(self.SampleWeights[train_index])
				
				hist = self.model.fit(Xt,yt,epochs=nEpoch,batch_size=BatchSize,validation_data=(Xc,yc),verbose=verbose,sample_weight=sw)

				self.Jt = np.append(self.Jt,hist.history['loss'])
				self.Jc = np.append(self.Jc,hist.history['val_loss'])
				self.hist.append(hist)
				k+=1
		return self.hist
		
	def Predict(self,X,RescaleY=True):
		if np.size(X.shape) == 1:
			x = np.array([X]).T
		else:
			x = X
		y = self.model.predict(x)
		if RescaleY:
			y = y/self.scale0 + self.scale1
		return y
	
	
	def GetWeights(self):
		w = []
		b = []
		tmp = self.model.get_weights()
		for i in range(0,self.L-1):
			w.append(tmp[i*2])
			b.append(tmp[i*2+1])
		return w,b
		
	def SetWeights(self,w,b):
		ipt = []
		for i in range(0,self.L-1):
			ipt.append(w[i])
			ipt.append(b[i])
		self.model.set_weights(ipt)
		
