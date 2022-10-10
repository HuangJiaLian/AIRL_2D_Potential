'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-11-20 21:03:15
@LastEditors: Jack Huang
@LastEditTime: 2019-11-20 21:18:06
'''
import numpy as np 
import matplotlib.pyplot as plt 
import os 

def main():
    data_path = 'data'
    if os.path.exists(data_path) != True:
        os.mkdir(data_path)
    
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
    
    # Draw y coordinate inverse
    # plt.gca().invert_yaxis()
    # plt.title('$V(x,y) = (1-x^2-y^2)^2 + y^2/(x^2+y^2)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    C = plt.contour(X, Y, Z, 25, colors='green')
    plt.scatter(-1,0,marker='x',color='tomato')
    plt.annotate('A=(-1,0)',(-1-offset,0-offset))
    plt.scatter(1,0,marker='x',color='tomato')
    plt.annotate('B=(1,0)',(1-offset,0-offset))
    plt.savefig('./{}/potential_surface_ry.png'.format(data_path))
    plt.show()

if __name__ == '__main__':
    main()