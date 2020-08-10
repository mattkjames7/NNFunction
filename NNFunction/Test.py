import numpy as np
import matplotlib.pyplot as plt
from .NNFunction import NNFunction
import os

def Func4(x):
	f0 = 0.1*x**3
	f1 = np.cos(x)
	f2 = np.arctan(x)
	f3 = 0.1*x**3 + 0.5*x**2 + x
	return np.array([f0,f1,f2,f3]).T
	
	
	
def _GetNN4():
	s = [1,4,6,4]

	model = NNFunction(s,'softplus')
	model.CreateData(Func4)
	
	return model	


def _GetAxis(fig,steps):
	ax0 = fig.subplot2grid((4,2),(0,0))
	ax1 = fig.subplot2grid((4,2),(1,0))
	ax2 = fig.subplot2grid((4,2),(0,1))
	ax3 = fig.subplot2grid((4,2),(1,1))
	ax4 = fig.subplot2grid((8,4),(4,1),rowspan=3,colspan=2)
	
	ax0.axis([-5,5,-1,1])
	ax1.axis([-5,5,-1,1])
	ax2.axis([-5,5,-1,1])
	ax3.axis([-5,5,-1,1])
	ax4.axis([-5,5,-1,1])
	
	ax0.set_xticks([])
	ax0.set_yticks([])
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax3.set_xticks([])
	ax3.set_yticks([])
	ax4.set_yticks([])
	
	ax4.set_ylabel('Cost')
	ax4.set_xlabel('Epochs')
	ax4.axis([0,steps,0.01,2.5])
	ax4.set_yscale('log')
	
	return ax0,ax1,ax2,ax3,ax4

def _PlotModels(ax0,ax1,ax2,ax3,model,x):
	p = model.Predict(x,False)
	ax0.scatter(x,model.y.T[0],color='red')
	ax0.plot(x,p.T[0],color='red',linestyle='-')	
	ax1.scatter(x,model.y.T[1],color='orange')
	ax1.plot(x,p.T[1],color='orange',linestyle='-')	
	ax2.scatter(x,model.y.T[2],color='green')
	ax2.plot(x,p.T[2],color='green',linestyle='-')	
	ax3.scatter(x,model.y.T[3],color='blue')
	ax3.plot(x,p.T[3],color='blue',linestyle='-')	

def TrainNN4(outpath='~/NNAnim/NN4/',steps=5000,savestep=100):
	
	if not os.path.isdir(outpath):
		os.system('mkdir -pv '+outpath)
	x = np.linspace(-5.0,5.0,100)
	model = _GetNN4()
	
	n = steps//savestep
		
	fig = plt
	fig.figure(figsize=(8,8))
	ax0,ax1,ax2,ax3,ax4 = _GetAxis(fig,steps)
	
	_PlotModels(ax0,ax1,ax2,ax3,model,x)
	fig.subplots_adjust(hspace=0.0,wspace=0.0)
	Epochs = 0
	Estr = 'Epochs: {:4d}'.format(Epochs)
	ax4.text(0.95,0.9,Estr,ha='right',va='center',transform=ax4.transAxes)
	
	fig.savefig(outpath+'NN{:05d}.png'.format(0))
	err = np.array([])
	for i in range(0,n):
		print('Frame {0} of {1}'.format(i+1,n))
		model.Train(savestep)
		fig.clf()
		ax0,ax1,ax2,ax3,ax4 = _GetAxis(fig,steps)
		_PlotModels(ax0,ax1,ax2,ax3,model,x)
		fig.subplots_adjust(hspace=0.0,wspace=0.0)
		Epochs = (i+1)*savestep
		Estr = 'Epochs: {:4d}'.format(Epochs)	
		ax4.text(0.95,0.9,Estr,ha='right',va='center',transform=ax4.transAxes)

		err = np.append(err,model.model.history.history['loss'])
		ax4.plot(err,color='grey')	
		fig.savefig(outpath+'NN{:05d}.png'.format(i+1))

def Func8(x):
	f0 = 2.0/(1.0 + np.exp(2*(x-3)))
	f1 = 0.1*x**3
	f2 = 0.1*x**3 + 0.5*x**2 + x
	f3 = np.sin(x)
	f4 = np.cos(x)
	f5 = np.arctan(x)
	f6 = np.exp(0.1*x)
	f7 = np.exp(0.1*x**2)
	return np.array([f0,f1,f2,f3,f4,f5,f6,f7]).T

def Test8FunctionFit(Hidden=[4,6],nEpoch=500):
	
	s = [1] + Hidden + [8]
	
	model = NNFunction(s,'softplus')
	model.CreateData(Func8)

	model.Train(nEpoch)
	
	plt.figure()
	colors = ['red','green','blue','black','orange','purple','cyan','gray']
	
	x = np.linspace(-5.0,5.0,100)
	p = model.Predict(x,False)
	for i in range(0,8):
		plt.plot(x,model.y.T[i],color=colors[i],linestyle='-')
		plt.plot(x,p.T[i],color=colors[i],linestyle='--')
	
	return model

def Test8FunctionFitPNN(Hidden=[4,6],nEpoch=1500):
	
	s = [1] + Hidden + [8]
	
	model = NNFunctionPNN(s,'softplus')
	model.CreateData(Func8)

	model.Train(nEpoch)
	
	plt.figure()
	colors = ['red','green','blue','black','orange','purple','cyan','gray']
	
	x = np.linspace(-5.0,5.0,100)
	p = model.Predict(x,False)
	for i in range(0,8):
		plt.plot(x,model.y.T[i],color=colors[i],linestyle='-')
		plt.plot(x,p[i],color=colors[i],linestyle='--')
		
	return model
