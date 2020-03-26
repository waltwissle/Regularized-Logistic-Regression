import numpy as np
import matplotlib.pyplot as plt
import time
import costFunctionreg, gradient, predict, mapFeature
import scipy.optimize as op


'''
This is regularized logistic regression to predict whether microchips from a 
fabrication plant passes quality assurance (QA). During QA, each microchip goes 
through various tests to ensure it is functioning correctly. 
Two test were conducted for some microchips. From these two tests, 
we would like to determine whether the microchips should be accepted or rejected. 
To help make the decision, the dataset of test results on past microchips are provided, 
from which we can build a logistic regression model.
Accept: Y = 1
Rejected: Y = 0
'''

## Load Data
data = np.loadtxt('data.txt', delimiter = ',')
X = data[:,0:2]; y = data[:,2]; Y = y.reshape(len(data),1)
pos = y == 1; neg = y== 0

print('--' * 44)
print('Plotting Training set  with + indicating (y = 1) samples and o indicating (y = 0) samples.')
print('--' * 44)

time.sleep(1.5) # pause for 1.5 secs

plt.figure()
plt.plot(X[pos,0],X[pos,1], 'k+',linewidth=12, markersize=7, label = 'y =1' ); 
plt.plot(X[neg,0],X[neg,1], 'ko', linewidth=2, markersize=6.5, mfc = 'r', label = 'y = 0'); 
plt.grid(linestyle ='--')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('Microchip Tests')
plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()

time.sleep(1.5) # pause for 1.5 secs

# =========== Regularized Logistic Regression ============
# Add Polynomial Features
print('Feature Mapping of the training Data\n');

phi = mapFeature.mapFeature(X[:,0], X[:,1]);

#Initialize fitting parameters
initial_theta = np.zeros((phi.shape[1],1))

# Set regularization parameter lambda to 1. 
l = 1;

cost  = costFunctionreg.costFunctionreg(initial_theta, phi, Y, l)
grad = gradient.gradient(initial_theta, phi, Y, l)

print(f'Cost at initial theta (zeros): \n{cost}\n');
print('Expected cost (approx): 0.693\n');
print('Gradient at initial theta (zeros) - first five values only:\n');
print(f'{grad[0:5]}\n', );
print('Expected gradients (approx) - first five values only:\n');
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');



# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones((phi.shape[1],1));
costs = costFunctionreg.costFunctionreg(test_theta, phi, Y, 10);
grads = gradient.gradient(test_theta, phi, Y, 10);

print(f'\nCost at test theta (with lambda = 10): \n {costs[0,0]}');
print('Expected cost (approx): 3.16\n');
print('Gradient at test theta - first five values only:\n');
print(f'\n{grads[0:5]}');
print('Expected gradients (approx) - first five values only:\n');
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

## ============= Part 2: Regularization and Accuracies =============
print('--' * 44)
print('Using Advanced Optimization techniques: "fmin_bfgs". ')
print('--' * 44)

l = 1 #lambda
initial_theta = np.zeros((phi.shape[1],1))


# Using the BFGS Method 
Result = op.fmin_bfgs(costFunctionreg.costFunctionreg, x0 = initial_theta, args = (phi, Y, l))


# Using the LBFGS Method 
Resultss= op.fmin_l_bfgs_b(costFunctionreg.costFunctionreg, x0 = initial_theta, args = (phi, Y, l), fprime = gradient.gradient)

# Seeting up parameters to plot the Decision boundary
u = np.linspace(-1,1.5,50).reshape((50,1))
v = np.linspace(-1,1.5,50).reshape((50,1))
z = np.zeros((len(u),len(v)))
theta = Result.reshape((phi.shape[1],1))

for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = mapFeature.mapFeature(u[i],v[j]) @ theta
 
print('--' * 44)
print('\n\nPlotting the Decision Boundary with lambda = 1 ')
print('--' * 44)        
 
plt.figure()
plt.plot(X[pos,0],X[pos,1], 'k+',linewidth=12, markersize=7, label = 'y =1' ); 
plt.plot(X[neg,0],X[neg,1], 'ko', linewidth=2, markersize=6.5, mfc = 'r')#, label = 'y = 0'); 
plt.contour(u.flatten(),v.flatten(),z.T,0, linewidths=2) #labels ='Deciscion Boundary')
plt.grid(linestyle ='--')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('Decision boundary with $\lambda$ = 1 ')
plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
#plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.show()
             
'''
X, Y : array-like, optional
The coordinates of the values in Z.

X and Y must both be 2-D with the same shape as Z (e.g. created via numpy.meshgrid), or they must both be 1-D such that len(X) == M is the number of columns in Z and len(Y) == N is the number of rows in Z.

If not given, they are assumed to be integer indices, i.e. X = range(M), Y = range(N).
'''

print('\n')
print('==' * 44)
print('\nCalculating the Accuracy of the Classifier\n')
p = predict.predict(theta, phi)
print(f'\nTrain Accuracy: {(np.mean(p==Y) * 100)}%')































