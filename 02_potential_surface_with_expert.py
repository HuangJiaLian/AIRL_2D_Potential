'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-10-14 21:56:59
@LastEditors: Jack Huang
@LastEditTime: 2019-11-22 15:37:53
'''
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os 

def main():
    data_path = 'data'
    env_data_path = 'env_data'
    if os.path.exists(env_data_path) != True:
        os.mkdir(env_data_path)
    
    x_min =  -1.2
    x_max = 1.2
    y_min = -0.2 
    y_max = 1.2

    offset = 0.07
    
    n = 100

    x = np.linspace(x_min,x_max,n)
    y = np.linspace(y_min,y_max,n)

    X,Y = np.meshgrid( x , y )
    Z = (1 - X**2 - Y**2)**2 +(Y**2)/(X**2 + Y**2)

    T = np.stack((X,Y,Z))

    np.save('./{}/potential.npy'.format(env_data_path),T)
    # plt.gca().invert_yaxis()
    # plt.title('$V(x,y) = (1-x^2-y^2)^2 + y^2/(x^2+y^2)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # Plot Potential contour
    # C = plt.contour(X, Y, Z, 25, colors='green')
    # Or use the following instead 
    C = plt.contour(T[0], T[1],T[2], 25, colors='green')
    plt.scatter(-1,0,marker='x',color='tomato')
    plt.annotate('A=(-1,0)',(-1-offset,0-offset))
    plt.scatter(1,0,marker='x',color='tomato')
    plt.annotate('B=(1,0)',(1-offset,0-offset))

    # Plot expert 
    df = pd.read_csv('./{}/expert_demo.csv'.format(env_data_path))
    ex_x = df.to_numpy()[:,0]
    ex_y = df.to_numpy()[:,1]
    plt.plot(ex_x,ex_y,color='tomato')

    plt.savefig('./{}/potential_surface_with_expert.png'.format(data_path))
    plt.show()

if __name__ == '__main__':
    main()