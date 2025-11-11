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

def forest_cover(C_crit, x, r, m, b, C, A):
    C = np.asarray(C)
    A = np.asarray(A)
    result = np.empty_like(C)
    mask = (1-x/r > calc_C_crit(A,m,b)) & (C>calc_C_crit(A,m,b))
    result[~mask] = -1
    result[mask] = 1
    return result


def calc_C_crit(A,m,b):
    return A*m+b

def calc_C_F(x,r):
    return(1-x/r)

def calc_A_F(x,r,m,b):
    return (calc_C_F(x,r)-b)/m


x=0.4
r=1.5
m=0.8
b=-0.1
resolution=int(1e3)
resolution_sparse=20
A=np.linspace(0,1.3,resolution)
A_sparse=np.linspace(0,1.3,resolution_sparse)
C=np.linspace(0,1.0,resolution)
C_sparse=np.linspace(0,1.0,resolution_sparse)
C_crit=calc_C_crit(A,m,b)
C_crit_sparse=calc_C_crit(A_sparse,m,b)

AA, CC = np.meshgrid(A, C)
AA_sparse, CC_sparse = np.meshgrid(A_sparse, C_sparse)

Direction=np.sign(forest_cover_derivative(C_crit,x,r,CC,AA))
State=forest_cover(C_crit,x,r,m,b,CC,AA)

Direction_sparse=(forest_cover_derivative(C_crit_sparse,x,r,CC_sparse,AA_sparse))

plt.pcolormesh(AA, CC,State, shading='auto', cmap='coolwarm_r')#, norm=norm)
plt.quiver(AA_sparse,CC_sparse, 0*np.ones_like(AA_sparse), Direction_sparse)
plt.plot(A,C_crit,color='white')
plt.plot([0,calc_A_F(x,r,m,b)],[calc_C_F(x,r),calc_C_F(x,r)],color='white')
plt.scatter([calc_A_F(x,r,m,b)],[calc_C_F(x,r)],color='white')
plt.plot([calc_A_F(x,r,m,b),calc_A_F(x,r,m,b)],[0,1],linestyle='--',color='white')
plt.plot([0,1.3],[calc_C_F(x,r),calc_C_F(x,r)],linestyle='--',color='white')
plt.ylim(0,1)
plt.savefig('results/test.png')