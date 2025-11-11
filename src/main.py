import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def forest_cover_derivative(C_crit, x, r, C, A):
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)
    
    mask = (C > C_crit)
    result[~mask] = -x * C[~mask]
    result[mask] = r * (1 - C[mask]) * C[mask] - x * C[mask]
    return result

def forest_cover(C_crit, x, r, C, A):
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)
    mask = (1-x/r > calc_C_crit(A)) & (C>calc_C_crit(A))
    result[~mask] = -1
    result[mask] = 1
    return result


def calc_C_crit(A):
    return A*0.8-0.1
resolution=int(1e3)
A=np.linspace(0,1.3,resolution)
C=np.linspace(0,1.0,resolution)
C_crit=calc_C_crit(A)
x=0.4
r=1.5
AA, CC = np.meshgrid(A, C)
State=forest_cover(C_crit,x,r,CC,AA)

plt.pcolormesh(AA, CC,State, shading='auto', cmap='coolwarm_r')#, norm=norm)

plt.savefig('results/test.png')