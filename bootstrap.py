import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

datos = np.loadtxt('notas_andes.dat')
x = datos[:,:4]
y = datos[:,4]


def betas(x,y):
	regression = sklearn.linear_model.LinearRegression()
	regression.fit(x,y)

	betas = np.array(())
	betas = np.append((regression.coef_),betas)

	return(np.array(betas))

beta_1 = np.array(())
beta_2 = np.array(())
beta_3 = np.array(())
beta_4 = np.array(())

for i in range(0,1000):
	a = np.linspace(0,np.shape(y)[0]-1,69, dtype = int)
	random = np.random.choice(a,np.shape(y)[0])
	x_rand = x[random]
	y_rand = y[random]

	betas_i = betas(x_rand,y_rand)

	beta_1 = np.append(betas_i[0],beta_1)
	beta_2 = np.append(betas_i[1],beta_2)
	beta_3 = np.append(betas_i[2],beta_3)
	beta_4 = np.append(betas_i[3],beta_4)

plt.subplot(2,2,1)
plt.hist(beta_1)
plt.subplot(2,2,2)
plt.hist(beta_2)
plt.subplot(2,2,3)
plt.hist(beta_3)
plt.subplot(2,2,4)
plt.hist(beta_4)

plt.show()

