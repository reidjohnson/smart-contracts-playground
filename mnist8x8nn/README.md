Implementation of a simple convolutional neural network to predict 8x8 pixel images of digits in a single Ethereum smart contract.

The model has been trained using PyTorch, and the trained model weights have then been manually quantized to integers and embedded into the contract. The model architecture is hardcoded into the contract, but the contract can optionally accept arrays specifying new model weights.

[Contract link](https://ropsten.etherscan.io/address/0xc135b522efe670ccb1466ea655c8dea6ac6a96cd) (Ropsten Test Network)
