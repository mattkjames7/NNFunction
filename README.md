# NNFunction
A simple package for modelling multidimensional non-linear functions using artificial neural networks.

## Installation

Install from `pip3`:

```bash
pip3 install --user NNFunction
```

Or by cloning this repository:

```bash
#clone the repo
git clone https://github.com/mattkjames7/NNFunction
cd NNFunction

#Either create a wheel and use pip: (X.X.X should be replaced with the current version)
python3 setup.py bdist_wheel
pip3 install --user dists/NNFunction-X.X.X-py3-none-any.whl

#Or by using setup.py directly
python3 setup.py install --user
```



## Usage

Start by training training a network:

```python
import NNFunction as nnf

#create the network, defining the activation functions and the number of nodes in each layer
net = nnf.NNFunction(s,AF='softplus',Output='linear')

#note that s should be a list, where each element denotes the number of nodes in each layer

#input training data
net.AddData(X,y)
#Input matrix X should be of the shape (m,n) - where m is the number of samples and n is the number of input features
#Output hypothesis matrix y should have the shape (m,k) - where k is the number of output nodes

#optionally add validation and test data
net.AddValidationData(Xv,yv)
#Note that validation data is ignored if kfolds > 1 during training
net.AddTestData(Xt,yt)

#Train the network 
net.Train(nEpoch,kfolds=k)
#nEpoch is the number of training epochs
#kfolds is the number of kfolds to do - if kfolds > 1 then the training data are split 
#into kfold sets, each of which has a turn at being the validation set. This results in
#kfold networks being trained in total (net.model)
#see docstring net.Train? to see more options

```

After training, the cost function may be plotted:

```python
net.PlotCost(k=k)
```

We can use the network on other data:

```python
#X in this case is a new matrix
y = net.Predict(X)
```

The networks can be saved and reloaded:

```python
#save
net.Save(fname='networkname.bin')

#reload
net = nnf.LoadANN(fname='networkname.bin')
```

The animation below demonstrates the training of a neural network used to reproduce four different functions simultaneously. It was produced using `NNFunction.TrainNN4`.

![](nn.gif)







