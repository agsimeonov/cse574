import numpy as np

def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-1.0 * z))

# Forward Pass

# Input
x_1 = .35
x_2 = .90

# Hidden
hidden_1 = (x_1 * .1) + (x_2 * .8)
hidden_1 = sigmoid(hidden_1)
hidden_2 = (x_1 * .4) + (x_2 * .6)
hidden_2 = sigmoid(hidden_2)

# Output
output = (hidden_1 * .3) + (hidden_2 * .9)
output = sigmoid(output)

# Backpropagation

# Output
error_output = (.5 - output) * output * (1 - output)
print("Error at output node:")
print(error_output)

# Hidden 1
error_1 = hidden_1 * (1 - hidden_1) * error_output * .3
print("Error at hidden node 1:")
print(error_1)
