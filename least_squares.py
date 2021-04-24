import numpy as np
import matplotlib.pyplot as plt

# create random data
x = np.random.randint(20,100,size=40)
y = x*0.7+np.random.randint(10, size=40)

# Least Square Method Linear regression
def LSM(x,y):
    # mean of x and y
    xm = sum(x)/len(x)
    ym = sum(y)/len(y)

    # xp = x - x mean 
    # yp = y - y mean
    # x2 = (x - x mean)^2
    # xy = (x - x mean)(y - y mean)
    xp,yp,x2,xy=[],[],[],[]

    for i in range(len(x)):
        xp.append(x[i]-xm)
        yp.append(y[i]-ym)
        x2.append(xp[i]**2)
        xy.append(xp[i]*yp[i])

    # y = b0 + b1*x
    # b1 is a slope and b0 is and intercept
    b1 = (sum(xy)/sum(x2))
    b0 = ym-b1*xm
    return b0,b1


# plotting data and prediction line
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot()
ax.scatter(x,y)
b0,b1 = LSM(x,y)
ax.plot([min(x),max(x)],[b1*min(x)+b0,b1*max(x)+b0])
plt.show()