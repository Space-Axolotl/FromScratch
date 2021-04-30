import numpy as np
import matplotlib.pyplot as plt

# create random data
x=np.random.randint(-10,10,size=40)
y= (x**2)+np.random.randint(-10,10,size=40)

# x2 = x^2
# xy = x*y
# x2y= (x^2)*y
x2,xy,x2y=[],[],[]

for i in range(len(x)):
    x2.append(x[i]**2)
    xy.append(x[i]*y[i])
    x2y.append((x[i]**2)*y[i])

x = np.array(x)
# create a matrix for calculations
A =      [[len(x),   sum(x),   sum(x**2)],
          [sum(x),   sum(x**2),sum(x**3)],
          [sum(x**2),sum(x**3),sum(x**4)]]
# take A-1 / A inverse
Ainv = np.linalg.inv(A)
# create vector with answers
B = [sum(y),sum(xy),sum(x2y)]
B = np.array(B)
# A*A-1 = I
# Ax=B /*A-1
# Ix = B*A-1
# x = B*A-1
# b here was represented as x
b = np.dot(Ainv, B.T)

# generate evenly spaced points to plot the function on
yp =[]
xp=np.arange(min(x),max(x),0.5)

for i in range(len(xp)):
    yp.append(b[0]+b[1]*xp[i]+b[2]*(xp[i]**2))

# plot the data with our calculated function
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot()
ax.scatter(x,y,c='mediumseagreen')
ax.plot(xp,yp,c='mediumblue')
plt.show()