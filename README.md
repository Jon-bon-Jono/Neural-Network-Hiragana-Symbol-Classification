# Neural-Network-Hiragana-Symbol-Classification
Linear function, fully connected (2 layer) and convolutional neural networks trained to classify handrawn old-style hiragana symbols (Kuzushiji) from the dataset Kuzushiji-MNIST. The classification tasks is inspired by the problem discussed in the paper: [Deep Learning for Classical Japanese Literature](https://arxiv.org/pdf/1812.01718.pdf).
## Usage
- Training and testing NetLin
  - ```kuzu_main.py --net lin```
- Training and testing NetFull
  - ```kuzu_main.py --net full```
- Training and testing NetConv
  - ```kuzu_main.py --net conv```
- Adjusting the learning rate
  - ```--lr```
- Adjusting the momentum
  - ```--mom```
- Adjusting the number of training epochs
  - ```--epochs```

## kuzu.py
- NetLin computes a linear function of the pixels in the image, followed by log softmax
- NetFull implements a fully connected 2-layer network using tanh at the hidden nodes and log softmax at the output node
- NetConv, has two convolutional layers plus one fully connected layer, all using relu activation function, followed by the output layer
  - Input is 1 channel with images of size 28x28 pixels. 
  - First convolutional layer has 10 filters of size 23x23 with padding of 1
  - Second convolutional layer has 27 filters of size 20x20 with padding of 2
  - Fully connected layer has 10800 neurons
  
