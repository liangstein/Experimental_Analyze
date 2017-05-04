from Tomography import DensityMatrix;import numpy as np;#加载Tomography库
from numpy import linalg as LA;import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
base=["HH","HV","VV","VH","RH","RV","DV","DH","DR","DD","RD","HD","VD",
                        "VL","HL","RL"]; #设置测量基顺序；
dm=DensityMatrix(base);
#counts=np.array([1912,6,1560,36,1082,771,1214,860,1085,1973,869,943,648,759,1067,25]);
counts=np.array([34749,324,35805,444,16324,17521,13441,16901,17932,32028,15132,17238,13171,
                17170,16722,33586]);#按序输入计数(文献PhysRevA.64.052312中的计数)
rho=dm.rho(counts);#根据计数直接得到的密度矩阵
rho_re=dm.rho_max_likelihood(counts,base);#最大似然估计后的密度矩阵(Tomography得到的态)
Concurrence=dm.concurrence(rho_re);#Tomography得到的concurrence
fidelity=dm.fidelity(rho_0,rho_re);#Tomography得到的密度矩阵和理想密度矩阵的fidelity
#a1,a2=LA.eig(rho_re);
real=np.zeros((4,4));image=np.zeros((4,4));
for i in np.arange(0,4):
    for j in np.arange(0,4):
        real[i,j]=rho[i,j].real;#获得密度矩阵的实部
        image[i,j]=rho[i,j].imag;#获得密度矩阵的虚部
