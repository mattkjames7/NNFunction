import numpy as np
import PyFileIO as pf
import os
from .NNFunction import NNFunction

def LoadANN(fname,ReturnModel=True,ReadCost=True):
	'''
	Read an ANN file
	
	
	Imports
	=======
	fname : str
		Name of the file to load from
	ReturnModel : bool
		If True, an ANN will be returned, otherwise only the weights and 
		costs
	ReadCost : bool
		If True, then the full cost arrays will be loaded with the ANN

		
	Returns
	=======
	if ReturnModel:
		NNFunction object (ANN)
	else:
		s,scale0,scale1,w,b,Jt,Jc
		
		s : number of layers
		scale0 : output parameter scale 
		scale1 : output parameter scale
		w : list of weight matrices
		b : list of bias matrices
		Jt : Training cost
		Jc : Validation cost
	
	'''
	

	print('Reading file: '+fname)
	#check the file exists
	if not os.path.isfile(fname):
		print('File not found')
		return None,None,None,None,None,np.array([]),np.array([])
		
	#load stuff from file
	f = open(fname,'rb')
	s = pf.ArrayFromFile('int32',f)
	scale0 = pf.ArrayFromFile('float32',f)
	scale1 = pf.ArrayFromFile('float32',f)
	w = pf.ListArrayFromFile('float32',f)
	b = pf.ListArrayFromFile('float32',f)
	if ReadCost:
		try:
			Jt = pf.ArrayFromFile('float32',f)
			Jc = pf.ArrayFromFile('float32',f)
		except:
			Jt = np.array([])
			Jc = np.array([])
	else:
		Jt = np.array([])
		Jc = np.array([])
	f.close()
	
	if ReturnModel:
		#create the neural network object
		model = NNFunction(s)

		model.k = 1
		model.model = [model._CreateModel()]
		model.Jt = [Jt]
		model.Jc = [Jc]
		model.hist = [[]]

		model.SetWeights(w,b)
		model.scale0 = scale0
		model.scale1 = scale1


		return model
	else:
		return s,scale0,scale1,w,b,Jt,Jc
