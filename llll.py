import numpy as np;from scipy.optimize import fsolve;
import matplotlib.pyplot as plt;
def transfer_matrix(theta_qwp1,theta_qwp2,theta_hwp):
    M_qwp1=np.matrix([[(np.cos(theta_qwp1))**2+1j*(np.sin(theta_qwp1))**2,(1-1j)*0.5*np.sin(2*theta_qwp1)],
                 [(1-1j)*0.5*np.sin(2*theta_qwp1),(np.sin(theta_qwp1))**2+1j*(np.cos(theta_qwp1))**2]]);
    M_qwp2=np.matrix([[(np.cos(theta_qwp2))**2+1j*(np.sin(theta_qwp2))**2,(1-1j)*0.5*np.sin(2*theta_qwp2)],
                 [(1-1j)*0.5*np.sin(2*theta_qwp2),(np.sin(theta_qwp2))**2+1j*(np.cos(theta_qwp2))**2]]);
    M_hwp=np.matrix([[np.cos(2*theta_hwp),np.sin(2*theta_hwp)],
                [np.sin(2*theta_hwp),-np.cos(2*theta_hwp)]]);
    return M_hwp*M_qwp1;

M=[[] for _ in np.arange(0,720)];#初始化总Jones矩阵
theta_qwp1_vol=np.linspace(0,np.pi,720);#1/4波片转角范围
theta_hwp_vol=np.linspace(0,np.pi,720);#半波片转角范围
for i in np.arange(0,720):
    for j in np.arange(0,720):
        mat=transfer_matrix(theta_qwp1_vol[i],0,theta_hwp_vol[j]);
        a=2**-0.5;
        #f=abs(mat[0,0]-1j*mat[1,1])**2+abs(mat[0,1]+1j*mat[1,0])**2+abs(mat[0,0]-mat[0,1])**2;#+x
        #f=abs(mat[0,0]+1j*mat[1,1])**2+abs(mat[0,1]-1j*mat[1,0])**2+abs(mat[0,0]+mat[0,1])**2;#-x
        #f=abs(mat[0,0]+1j*mat[1,1])**2+abs(mat[0,1]+1j*mat[1,0])**2+abs(mat[0,0]-1j*mat[0,1])**2;#+y
        f=abs(mat[0,0]-1j*mat[1,1])**2+abs(mat[0,1]-1j*mat[1,0])**2+abs(mat[0,0]+1j*mat[0,1])**2;#-y
        M[i].append(f);#建立总Jones矩阵

#plt.imshow(np.asarray(M),extent=[-1,1,1,-1],cmap="jet");plt.colorbar();
#plt.show()

min=10;#极小值初值
for i in np.arange(0,720):
    for j in np.arange(0,720):
        if min > M[i][j]:
            theta_qwp1=theta_qwp1_vol[i];
            theta_hwp=theta_hwp_vol[j];
            pos_i=i;pos_j=j;min=M[i][j];#拟合目标Jones矩阵

theta_qwp1*np.pi**-1*180;theta_hwp*np.pi**-1*180;#得到的1/4和1/2波片角度
transfer_matrix(theta_qwp1,0,theta_hwp);#拟合后的总Jones矩阵


from sympy import *;import numpy as np;import mpmath;
mpmath.mp.dps=20;
theta_qwp1=Symbol("theta_qwp1",real=True);
theta_qwp2=Symbol("theta_qwp2",real=True);
theta_hwp=Symbol("theta_hwp",real=True);
M_qwp1=Matrix([[(cos(theta_qwp1))**2+1j*(sin(theta_qwp1))**2,(1-1j)*sin(theta_qwp1)*cos(theta_qwp1)],
               [(1-1j)*sin(theta_qwp1)*cos(theta_qwp1),(sin(theta_qwp1))**2+1j*(cos(theta_qwp1))**2]]);
M_qwp2=Matrix([[(cos(theta_qwp2))**2+1j*(sin(theta_qwp2))**2,(1-1j)*sin(theta_qwp2)*cos(theta_qwp2)],
               [(1-1j)*sin(theta_qwp2)*cos(theta_qwp2),(sin(theta_qwp2))**2+1j*(cos(theta_qwp2))**2]]);
M_hwp=Matrix([[cos(2*theta_hwp),sin(2*theta_hwp)],
              [sin(2*theta_hwp),-cos(2*theta_hwp)]]);
M=simplify(M_hwp*M_qwp1);#Jones矩阵的形式推导
