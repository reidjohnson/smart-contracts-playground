# MNIST8x8NeuralNet

Implementation of a simple convolutional neural network to predict 8x8 pixel images of digits in a single Ethereum smart contract.

The model has been trained using PyTorch on MNIST images resized to 8x8 pixels. The trained model weights have then been manually quantized to integers and embedded into the contract along with an example input image. The model architecture is hardcoded into the contract, but the contract can optionally accept arrays specifying a new input image and/or new model weights.

[Contract link](https://ropsten.etherscan.io/address/0x599505d6c0d2d5306438b69a0db2d8af53886a83) (Ropsten Test Network)
