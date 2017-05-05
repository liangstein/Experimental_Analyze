import numpy as np;from scipy.optimize import minimize;
from numpy import linalg as LA;import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
base=["HH","HV","VV","VH","RH","RV","DV","DH","DR","DD","RD","HD","VD","VL","HL","RL"]; #设置测量基顺序；
#rho_0=0.5*np.array([[],[],[],[]]);
#n=np.array([1912,6,1560,36,1082,771,1214,860,1085,1973,869,943,648,759,1067,25]);
n=np.array([34749,324,35805,444,16324,17521,13441,16901,17932,32028,15132,17238,13171,
                17170,16722,33586]);#按序输入计数(文献PhysRevA.64.052312中的计数)
N=(n[0]+n[1]+n[2]+n[3]);
H=np.array([1,0]);V=np.array([0,1]);
R=2**-0.5*np.array([1,1j]);D=2**-0.5*np.array([1,1]);L=2**-0.5*np.array([1,-1j]);
base1=np.kron(H,H);base2=np.kron(H,V);base3=np.kron(V,V);base4=np.kron(V,H);
base5=np.kron(R,H);base6=np.kron(R,V);base7=np.kron(D,V);base8=np.kron(D,H);
base9=np.kron(D,R);base10=np.kron(D,D);base11=np.kron(R,D);base12=np.kron(H,D);
base13=np.kron(V,D);base14=np.kron(V,L);base15=np.kron(H,L);base16=np.kron(R,L);
def concurrence(rho):
    a1,a2=LA.eig(rho);
    a1=sorted(a1);
    C=max(0,a1[-1]-a1[-2]-a1[-3]-a1[-4]);
    return C;

def rho(t):
    T=np.matrix([[t[0],0,0,0],
                      [t[4]+1j*t[5],t[1],0,0],
                      [t[10]+1j*t[11],t[6]+1j*t[7],t[2],0],
                      [t[14]+1j*t[15],t[12]+1j*t[13],t[8]+1j*t[9],t[3]]]);
    rho0=T.getH()*T;
    dm=rho0*(rho0[0,0].real+rho0[1,1].real+rho0[2,2].real+rho0[3,3].real)**-1;
    return dm;

def Log_likelihood(t):
    dm=rho(t);
    K1=float(((base1*dm).dot(base1.conjugate())).real);
    K2 = float(((base2 * dm).dot(base2.conjugate())).real);
    K3 = float(((base3 * dm).dot(base3.conjugate())).real);
    K4 = float(((base4 * dm).dot(base4.conjugate())).real);
    K5 = float(((base5 * dm).dot(base5.conjugate())).real);
    K6 = float(((base6 * dm).dot(base6.conjugate())).real);
    K7 = float(((base7 * dm).dot(base7.conjugate())).real);
    K8 = float(((base8 * dm).dot(base8.conjugate())).real);
    K9 = float(((base9 * dm).dot(base9.conjugate())).real);
    K10 = float(((base10 * dm).dot(base10.conjugate())).real);
    K11 = float(((base11 * dm).dot(base11.conjugate())).real);
    K12 = float(((base12 * dm).dot(base12.conjugate())).real);
    K13 = float(((base13 * dm).dot(base13.conjugate())).real);
    K14 = float(((base14 * dm).dot(base14.conjugate())).real);
    K15 = float(((base15 * dm).dot(base15.conjugate())).real);
    K16 = float(((base16 * dm).dot(base16.conjugate())).real);
    L1=(N*K1-n[0])**2*(N*K1)**-1;L2=(N*K2-n[1])**2*(N*K2)**-1;
    L3 = (N * K3 - n[2]) ** 2 * (N * K3) ** -1;
    L4 = (N * K4 - n[3]) ** 2 * (N * K4) ** -1;
    L5 = (N * K5 - n[4]) ** 2 * (N * K5) ** -1;
    L6 = (N * K6 - n[5]) ** 2 * (N * K6) ** -1;
    L7 = (N * K7 - n[6]) ** 2 * (N * K7) ** -1;
    L8 = (N * K8 - n[7]) ** 2 * (N * K8) ** -1;
    L9 = (N * K9 - n[8]) ** 2 * (N * K9) ** -1;
    L10 = (N * K10 - n[9]) ** 2 * (N * K10) ** -1;
    L11 = (N * K11 - n[10]) ** 2 * (N * K11) ** -1;
    L12 = (N * K12 - n[11]) ** 2 * (N * K12) ** -1;
    L13 = (N * K13 - n[12]) ** 2 * (N * K13) ** -1;
    L14 = (N * K14 - n[13]) ** 2 * (N * K14) ** -1;
    L15 = (N * K15 - n[14]) ** 2 * (N * K15) ** -1;
    L16 = (N * K16 - n[15]) ** 2 * (N * K16) ** -1;
    return L1+L2+L3+L4+L5+L6+L7+L8+L9+L10+L11+L12+L13+L14+L15+L16;

t0=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.1,0.1,0.2,0.3,0.1,0.2,0.3,0.1];#t序列初始值
res=minimize(Log_likelihood,t0,method='BFGS');
rho_re=rho(res.x);#重建后的密度矩阵
a1,a2=LA.eig(rho_re);
real=np.zeros((4,4));image=np.zeros((4,4));
for i in np.arange(0,4):
    for j in np.arange(0,4):
        real[i,j]=rho_re[i,j].real;
        image[i,j]=rho_re[i,j].imag;

concurrence(rho_re);
#array([[ 0.50322366, -0.02077412, -0.02437523,  0.46619939],
#       [-0.02077412,  0.00520623,  0.00406566, -0.03202184],
#       [-0.02437523,  0.00406566,  0.00723096, -0.03918401],
#       [ 0.46619939, -0.03202184, -0.03918401,  0.48433914]])
#array([[ 0.        ,  0.01105538, -0.01850889,  0.02195773],
#       [-0.01105538,  0.        , -0.00186784, -0.0051833 ],
#       [ 0.01850889,  0.00186784,  0.        ,  0.01089078],
#       [-0.02195773,  0.0051833 , -0.01089078,  0.        ]])
#本征值 [  9.64746954e-01 -5.05847018e-18j,
#         3.52530461e-02 +2.72615217e-18j,
#         8.74282558e-13 -2.94774485e-18j,   1.97487594e-10 -1.50026549e-18j]