# from sympy import symbols, diff
# >>> x, y = symbols('x y', real=True)
# >>> diff( x**2 + y**3, y)
# 3*y**2
# >>> diff( x**2 + y**3, y).subs({x:3, y:1})


#-*- coding:utf-8 -*-
import numpy as np
from sympy import symbols, diff
import csv
from matplotlib import pyplot as plt
csv_data = []
with open('train_data.csv', 'r') as f:
    reader = csv.reader(f)
    print(type(reader))
    for row in reader:
         csv_data.append(row)
csv_data = csv_data[1:-1]
print(csv_data)


def least_square(csv_data):
    txy, tx, ty, tx2, txy_mul, tx2_mul, tx_2 = 0, 0, 0, 0, 0, 0, 0
    for i in range(len(csv_data)):
        txy = txy + int(csv_data[i][0])*int(csv_data[i][1]) #x*y
        tx = tx + int(csv_data[i][0]) #x
        ty = ty + int(csv_data[i][1]) #y
        tx2 = tx2 + int(csv_data[i][0])**2 #x**2
    txy = len(csv_data) * txy
    txy_mul = tx * ty
    tx2_mul = len(csv_data) * tx2
    tx_2 = tx**2
    k = (txy - txy_mul) / (tx2_mul - tx_2)
    b = ty / len(csv_data) - k * tx / len(csv_data)
    return k, b


if __name__ == '__main__':
    x = np.arange(1, 3, 1)
    k, b = least_square(csv_data)
    y = k * x + b
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("yiyuanyicihanshu")
    plt.plot(x, y)
    highs = [(1, 6), (2, 5), (3, 7)]
    fig = plt.figure(dpi=128)
    plt.plot(highs, c='red')
    plt.show()
