import random
import math

# Initializing values for the weights between -1.2 and 1.2
W13 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]
W14 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]
W23 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]
W24 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]
W35 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]
W45 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]

# Initializing arrays to hold initial values
iW13 = [0, 0, 0, 0]
iW14 = [0, 0, 0, 0]
iW23 = [0, 0, 0, 0]
iW24 = [0, 0, 0, 0]
iW35 = [0, 0, 0, 0]
iW45 = [0, 0, 0, 0]
theta3 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]
theta4 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]
theta5 = [(random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10, (random.randrange(-12, 12))/10]

# More arrays initialized
# Output for each state
Y3 = [0, 0, 0, 0]
Y4 = [0, 0, 0, 0]
Y5 = [0, 0, 0, 0]
# Error for each state
Error = [0, 0, 0, 0]
# Desired output: what you want to train your NN to solve
Yd = [0, 1, 1, 0]                               

d3 = [0, 0, 0, 0]
d4 = [0, 0, 0, 0]
d5 = [0, 0, 0, 0]

dW13 = [0, 0, 0, 0]
dW14 = [0, 0, 0, 0]
dW23 = [0, 0, 0, 0]
dW24 = [0, 0, 0, 0]
dW35 = [0, 0, 0, 0]
dW45 = [0, 0, 0, 0]
dW13p = [0, 0, 0, 0]
dW14p = [0, 0, 0, 0]
dW23p = [0, 0, 0, 0]
dW24p = [0, 0, 0, 0]
dW35p = [0, 0, 0, 0]
dW45p = [0, 0, 0, 0]

dtheta3 = [0, 0, 0, 0]
dtheta4 = [0, 0, 0, 0]
dtheta5 = [0, 0, 0, 0]

alpha = 0.1
beta = 0.95
a = 1.716                                        
b = 0.667
soqe = 1
psoqe = 1
n1 = 0
n2 = 0
epoch = 1

# Function definitions
# Sigmoid equation
def sigmoid(x1, w1, x2, w2, t):
    exp = ((x1 * w1) - t) + ((x2 * w2) - t)
    return (1/(1 + (math.e ** -exp)))

# Hyperbolic tangent equation
def hyperbolic_tangent(x1, w1, x2, w2, t):
    exp = ((x1 * w1) - t) + ((x2 * w2) - t)
    if exp > 0:
        return (((2 * a) / (1 + (math.e ** -(b * exp)))) - a)
    else:
        return (((2 * a) / (1 + (math.e ** (b * exp)))) - a)

# Error calculation
def error(y, yd):
    return (yd - y)

# Error gradient calculation
def error_gradient(y, d, w):
    return (y * (1 - y) * d * w)

# Weight correctness calculation
def weight_correctness(x, d):
    return (alpha * x * d)

def update(a, b):
    return (a + b)

# Sum of squared errors calculation
def sum_of_squared_errors(e0, e1, e2, e3):
    return (((e0 - 0) ** 2) + ((e1 - 0) ** 2) + ((e2 - 0) ** 2) + ((e3 - 0) ** 2))

while soqe >= 0.001:                                                 # Runs until specific sum of squared errors
    iW13 = W13
    iW14 = W14
    iW23 = W23
    iW24 = W24
    iW35 = W35
    iW45 = W45 

    # 0,0
    Y3[0] = hyperbolic_tangent(0, W13[0], 0, W23[0], theta3[0])      # Calculates outputs
    Y4[0] = hyperbolic_tangent(0, W14[0], 0, W24[0], theta4[0])
    Y5[0] = sigmoid(Y3[0], W35[0], Y4[0], W45[0], theta5[0])
    Error[0] = round(error(Y5[0], Yd[0]), 7)                         # Calculates percent error
    d5[0] = (Y5[0] * (1 - Y5[0])) * Error[0]                         # Calculates error gradient
    d4[0] = error_gradient(Y4[0], d5[0], W45[0])
    d3[0] = error_gradient(Y3[0], d5[0], W35[0])
    dW13[0] = (beta * dW13p[0]) + weight_correctness(0, d3[0])        # Calculates weight correctness
    dW23[0] = (beta * dW23p[0]) + weight_correctness(0, d3[0])
    dW14[0] = (beta * dW14p[0]) + weight_correctness(0, d4[0])
    dW24[0] = (beta * dW24p[0]) + weight_correctness(0, d4[0])
    dW35[0] = (beta * dW35p[0]) + weight_correctness(Y3[0], d5[0])
    dW45[0] = (beta * dW45p[0]) + weight_correctness(Y4[0], d5[0])
    dW13p[0] = weight_correctness(0, d3[0])
    dW23p[0] = weight_correctness(0, d3[0])
    dW14p[0] = weight_correctness(0, d4[0])
    dW24p[0] = weight_correctness(0, d4[0])
    dW35p[0] = weight_correctness(Y3[0], d5[0])
    dW45p[0] = weight_correctness(Y4[0], d5[0])
    dtheta3[0] = weight_correctness(-1, d3[0])
    dtheta4[0] = weight_correctness(-1, d4[0])
    dtheta5[0] = weight_correctness(-1, d5[0])
    W13[0] = update(W13[0], dW13[0])                                 # Updates the current weights
    W14[0] = update(W14[0], dW14[0])
    W23[0] = update(W23[0], dW23[0])
    W24[0] = update(W24[0], dW24[0])
    W35[0] = update(W35[0], dW35[0])
    W45[0] = update(W45[0], dW45[0])
    theta3[0] = update(theta3[0], dtheta3[0])
    theta4[0] = update(theta4[0], dtheta4[0])
    theta5[0] = update(theta5[0], dtheta5[0])

    # 1,0
    Y3[1] = hyperbolic_tangent(0, W13[1], 1, W23[1], theta3[1])
    Y4[1] = hyperbolic_tangent(0, W14[1], 1, W24[1], theta4[1])
    Y5[1] = sigmoid(Y3[1], W35[1], Y4[1], W45[1], theta5[1])
    Error[1] = round(error(Y5[1], Yd[1]), 7)
    d5[1] = (Y5[1] * (1 - Y5[1])) * Error[1]
    d4[1] = error_gradient(Y4[1], d5[1], W45[1])
    d3[1] = error_gradient(Y3[1], d5[1], W35[1])
    dW13[1] = (beta * dW13p[1]) + weight_correctness(0, d3[1])
    dW23[1] = (beta * dW23p[1]) + weight_correctness(1, d3[1])
    dW14[1] = (beta * dW14p[1]) + weight_correctness(0, d4[1])
    dW24[1] = (beta * dW24p[1]) + weight_correctness(1, d4[1])
    dW35[1] = (beta * dW35p[1]) + weight_correctness(Y3[1], d5[1])
    dW45[1] = (beta * dW45p[1]) + weight_correctness(Y4[1], d5[1])
    dW13p[1] = weight_correctness(0, d3[1])
    dW23p[1] = weight_correctness(1, d3[1])
    dW14p[1] = weight_correctness(0, d4[1])
    dW24p[1] = weight_correctness(1, d4[1])
    dW35p[1] = weight_correctness(Y3[1], d5[1])
    dW45p[1] = weight_correctness(Y4[1], d5[1])
    dtheta3[1] = weight_correctness(-1, d3[1])
    dtheta4[1] = weight_correctness(-1, d4[1])
    dtheta5[1] = weight_correctness(-1, d5[1])
    W13[1] = update(W13[1], dW13[1])
    W14[1] = update(W14[1], dW14[1])
    W23[1] = update(W23[1], dW23[1])
    W24[1] = update(W24[1], dW24[1])
    W35[1] = update(W35[1], dW35[1])
    W45[1] = update(W45[1], dW45[1])
    theta3[1] = update(theta3[1], dtheta3[1])
    theta4[1] = update(theta4[1], dtheta4[1])
    theta5[1] = update(theta5[1], dtheta5[1])

    # 0,1
    Y3[2] = hyperbolic_tangent(1, W13[2], 0, W23[2], theta3[2])
    Y4[2] = hyperbolic_tangent(1, W14[2], 0, W24[2], theta4[2])
    Y5[2] = sigmoid(Y3[2], W35[2], Y4[2], W45[2], theta5[2])
    Error[2] = round(error(Y5[2], Yd[2]), 7)
    d5[2] = (Y5[2] * (1 - Y5[2])) * Error[2]
    d4[2] = error_gradient(Y4[2], d5[2], W45[2])
    d3[2] = error_gradient(Y3[2], d5[2], W35[2])
    dW13[2] = (beta * dW13p[2]) + weight_correctness(1, d3[2])
    dW23[2] = (beta * dW23p[2]) + weight_correctness(0, d3[2])
    dW14[2] = (beta * dW14p[2]) + weight_correctness(1, d4[2])
    dW24[2] = (beta * dW24p[2]) + weight_correctness(0, d4[2])
    dW35[2] = (beta * dW35p[2]) + weight_correctness(Y3[2], d5[2])
    dW45[2] = (beta * dW45p[2]) + weight_correctness(Y4[2], d5[2])
    dW13p[2] = weight_correctness(1, d3[2])
    dW23p[2] = weight_correctness(0, d3[2])
    dW14p[2] = weight_correctness(1, d4[2])
    dW24p[2] = weight_correctness(0, d4[2])
    dW35p[2] = weight_correctness(Y3[2], d5[2])
    dW45p[2] = weight_correctness(Y4[2], d5[2])
    dtheta3[2] = weight_correctness(-1, d3[2])
    dtheta4[2] = weight_correctness(-1, d4[2])
    dtheta5[2] = weight_correctness(-1, d5[2])
    W13[2] = update(W13[2], dW13[2])
    W14[2] = update(W14[2], dW14[2])
    W23[2] = update(W23[2], dW23[2])
    W24[2] = update(W24[2], dW24[2])
    W35[2] = update(W35[2], dW35[2])
    W45[2] = update(W45[2], dW45[2])
    theta3[2] = update(theta3[2], dtheta3[2])
    theta4[2] = update(theta4[2], dtheta4[2])
    theta5[2] = update(theta5[2], dtheta5[2])

    # 1,1
    Y3[3] = hyperbolic_tangent(1, W13[3], 1, W23[3], theta3[3])
    Y4[3] = hyperbolic_tangent(1, W14[3], 1, W24[3], theta4[3])
    Y5[3] = sigmoid(Y3[3], W35[3], Y4[3], W45[3], theta5[3])
    Error[3] = round(error(Y5[3], Yd[3]), 7)
    d5[3] = (Y5[3] * (1 - Y5[3])) * Error[3]
    d4[3] = error_gradient(Y4[3], d5[3], W45[3])
    d3[3] = error_gradient(Y3[3], d5[3], W35[3])
    dW13[3] = (beta * dW13p[3]) + weight_correctness(1, d3[3])
    dW23[3] = (beta * dW23p[3]) + weight_correctness(1, d3[3])
    dW14[3] = (beta * dW14p[3]) + weight_correctness(1, d4[3])
    dW24[3] = (beta * dW24p[3]) + weight_correctness(1, d4[3])
    dW35[3] = (beta * dW35p[3]) + weight_correctness(Y3[0], d5[3])
    dW45[3] = (beta * dW45p[3]) + weight_correctness(Y4[0], d5[3])
    dW13p[3] = weight_correctness(1, d3[3])
    dW23p[3] = weight_correctness(1, d3[3])
    dW14p[3] = weight_correctness(1, d4[3])
    dW24p[3] = weight_correctness(1, d4[3])
    dW35p[3] = weight_correctness(Y3[0], d5[3])
    dW45p[3] = weight_correctness(Y4[0], d5[3])
    dtheta3[3] = weight_correctness(-1, d3[3])
    dtheta4[3] = weight_correctness(-1, d4[3])
    dtheta5[3] = weight_correctness(-1, d5[3])
    W13[3] = update(W13[3], dW13[3])
    W14[3] = update(W14[3], dW14[3])
    W23[3] = update(W23[3], dW23[3])
    W24[3] = update(W24[3], dW24[3])
    W35[3] = update(W35[3], dW35[3])
    W45[3] = update(W45[3], dW45[3])
    theta3[3] = update(theta3[3], dtheta3[3])
    theta4[3] = update(theta4[3], dtheta4[3])
    theta5[3] = update(theta5[3], dtheta5[3])

    psoqe = soqe
    soqe = sum_of_squared_errors(Error[0], Error[1], Error[2], Error[3])

    if (soqe > 0 and psoqe > 0) or (soqe < 0 and psoqe < 0):
        n1 += 1
        n2 = 0
    elif (soqe > 0 and psoqe < 0) or (soqe < 0 and psoqe > 0):
        n2 += 1
        n1 = 0
    if n1 > 3:
        n1 = 0
        n2 = 0
        alpha *= 1.1
    elif n2 > 3:
        n1 = 0
        n2 = 0
        alpha *= 0.7

    epoch += 1
    # Continuelly prints a chart showing the results from each epoch
    print("-------+---------+------------+-----------------------------------------------+--------+--------+----------------------------------------------")
    print(" Epoch | Inputs  | Desired    | Initial Weights                               | Actual | Error  | Final Weights")
    print("       | X1 | X2 | output: Yd |  W13  |  W23  |  W14  |  W24  |  W35  |  W45  | out:Y  | e      |  W13  |  W23  |  W14  |  W24  |  W35  |  W45  ")
    print("-------+----+----+------------+-------+-------+-------+-------+-------+-------+--------+--------+-------+-------+-------+-------+-------+------")
    print("       | 1  | 1  |    ", Yd[3], "     |",'{: 3.1f}'.format(round(iW13[3], 1))," |",'{: 3.1f}'.format(round(iW23[3], 1)), " |",'{: 3.1f}'.format(round(iW14[3], 1))," |",'{: 3.1f}'.format(round(iW24[3], 1))," |", '{: 3.1f}'.format(round(iW35[3], 1))," |",'{: 3.1f}'.format(round(iW45[3], 1))," |","{: .3F}".format(round(Y5[3], 3)), "|", "{: .3F}".format(round(Error[3], 3)), "|", '{: 3.1f}'.format(round(W13[3], 1)), " |", '{: 3.1f}'.format(round(W23[3], 1)), " |", '{: 3.1f}'.format(round(W14[3], 1)), " |", '{: 3.1f}'.format(round(W24[3], 1)), " |", '{: 3.1f}'.format(round(W35[3], 1)), " |", '{: 3.1f}'.format(round(W45[3], 1)))
    print(" ",'{:3d}'.format(epoch), " | 0  | 1  |    ", Yd[2], "     |",'{: 3.1f}'.format(round(iW13[2], 1))," |",'{: 3.1f}'.format(round(iW23[2], 1))," |", '{: 3.1f}'.format(round(iW14[2], 1))," |",'{: 3.1f}'.format(round(iW24[2], 1))," |",'{: 3.1f}'.format(round(iW35[2], 1))," |",'{: 3.1f}'.format(round(iW45[2], 1))," |","{: .3F}".format(round(Y5[2], 3)), "|", "{: .3F}".format(round(Error[2], 3)), "|", '{: 3.1f}'.format(round(W13[2], 1)), " |", '{: 3.1f}'.format(round(W23[2], 1)), " |", '{: 3.1f}'.format(round(W14[2], 1)), " |", '{: 3.1f}'.format(round(W24[2], 1)), " |", '{: 3.1f}'.format(round(W35[2], 1)), " |", '{: 3.1f}'.format(round(W45[2], 1)))
    print("       | 1  | 0  |    ", Yd[1], "     |",'{: 3.1f}'.format(round(iW13[1], 1))," |",'{: 3.1f}'.format(round(iW23[1], 1))," |", '{: 3.1f}'.format(round(iW14[1], 1))," |",'{: 3.1f}'.format(round(iW24[1], 1))," |",'{: 3.1f}'.format(round(iW35[1], 1))," |",'{: 3.1f}'.format(round(iW45[1], 1))," |","{: .3F}".format(round(Y5[1], 3)), "|", "{: .3F}".format(round(Error[1], 3)), "|", '{: 3.1f}'.format(round(W13[1], 1)), " |", '{: 3.1f}'.format(round(W23[1], 1)), " |", '{: 3.1f}'.format(round(W14[1], 1)), " |", '{: 3.1f}'.format(round(W24[1], 1)), " |", '{: 3.1f}'.format(round(W35[1], 1)), " |", '{: 3.1f}'.format(round(W45[1], 1)))
    print("       | 0  | 0  |    ", Yd[0], "     |",'{: 3.1f}'.format(round(iW13[0], 1))," |",'{: 3.1f}'.format(round(iW23[0], 1))," |", '{: 3.1f}'.format(round(iW14[0], 1))," |",'{: 3.1f}'.format(round(iW24[0], 1))," |",'{: 3.1f}'.format(round(iW35[0], 1))," |",'{: 3.1f}'.format(round(iW45[0], 1))," |","{: .3F}".format(round(Y5[0], 3)), "|", "{: .3F}".format(round(Error[0], 3)), "|", '{: 3.1f}'.format(round(W13[0], 1)), " |", '{: 3.1f}'.format(round(W23[0], 1)), " |", '{: 3.1f}'.format(round(W14[0], 1)), " |", '{: 3.1f}'.format(round(W24[0], 1)), " |", '{: 3.1f}'.format(round(W35[0], 1)), " |", '{: 3.1f}'.format(round(W45[0], 1)))
# Prints a final chart showing more consolidated numbers
print("---------------------------------------------------------------------------------------------------------------")
print(" Epoch   | X1    | X2    | Desired output Yd | Actual output   Y     | Error      e    | Sum of Squared Errors")
print("---------+-------+-------+-------------------+-----------------------+-----------------+-----------------------")
print("         | 1     | 1     |     ", Yd[3], "           | ", "{: .7F}".format(Y5[3]), "          | ", "{: .7F}".format(Error[3]), "    | ", "{:.5f}".format(round(soqe, 5)))
print(" ",'{:3d}'.format(epoch), "   | 0     | 1     |     ", Yd[2], "           | ", "{: .7F}".format(Y5[2]), "          | ", "{: .7F}".format(Error[2]), "    |")
print("         | 1     | 0     |     ", Yd[1], "           | ", "{: .7F}".format(Y5[1]), "          | ", "{: .7F}".format(Error[1]), "    |")
print("         | 0     | 0     |     ", Yd[0], "           | ", "{: .7F}".format(Y5[0]), "          | ", "{: .7F}".format(Error[0]), "    |")