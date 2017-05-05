import numpy as np;from scipy.optimize import minimize;from Tomography import DensityMatrix;
rho_0=0.5*np.matrix([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]]);
#特例
A=np.matrix([[0,1],[1,0]]);
B=2**-0.5*np.matrix([[0,1-1j],[1+1j,0]]);
D=np.matrix([[0,-1j],[1j,0]]);
C=2**-0.5*np.matrix([[0,-1-1j],[-1+1j,0]]);
def trace(input):
    return input[0,0]+input[1,1]+input[2,2]+input[3,3];

P_ab=trace(rho_0*np.kron(A,B));P_ac=trace(rho_0*np.kron(A,C));
P_dc=trace(rho_0*np.kron(D,C));P_db=trace(rho_0*np.kron(D,B));
CHSH=abs(P_ab-P_ac)+abs(P_dc+P_db);
#通解
sigma_x=np.array([[0,1],[1,0]]);sigma_y=np.array([[0,-1j],[1j,0]]);
def trace(input):
    return input[0,0]+input[1,1]+input[2,2]+input[3,3];

def minus_CHSH(x):
    theta_a=x[0];theta_b=x[1];theta_c=x[2];theta_d=x[3];
    A=sigma_x*np.cos(theta_a)+sigma_y*np.sin(theta_a);
    B=sigma_x*np.cos(theta_b)+sigma_y*np.sin(theta_b);
    C=sigma_x*np.cos(theta_c)+sigma_y*np.sin(theta_c);
    D=sigma_x*np.cos(theta_d)+sigma_y*np.sin(theta_d);
    P_ab = trace(rho_0 * np.kron(A, B));
    P_ac = trace(rho_0 * np.kron(A, C));
    P_dc = trace(rho_0 * np.kron(D, C));
    P_db = trace(rho_0 * np.kron(D, B));
    return -abs(P_ab - P_ac) - abs(P_dc + P_db);

x0=[0.1,0.2,0.3,0.4];#初始角度
res=minimize(minus_CHSH,x0);#最小化解极值
res.x;#最大CHSH违背时对应的投影角度
abs(minus_CHSH(res.x));#最大CHSH违背时对应的CHSH

#找到的一组投影角度(弧度)是 [ 3.53429415,  0.3927037 ,  1.96348998, -1.17810025], CHSH极值2.8284271246994486

#now we calculate CHSH by collected numbers
#            +         -        L        R
# QWP/HWP  45/22.5  45/67.5  90/22.5  0/22.5
basis=["++","+-","+L","+R","-+","--","-L","-R","L+","L-","LL","LR","R+","R-","RL","RR"];
counts=[];
P_ab=2**-0.5*(counts[0]-counts[1]+counts[2]-counts[3]-counts[4]+counts[5]-counts[6]+counts[7]);
P_ac=2**-0.5*(-counts[0]+counts[1]+counts[2]-counts[3]+counts[4]-counts[5]-counts[6]+counts[7]);
P_db=2**-0.5*(counts[8]-counts[9]+counts[10]-counts[11]-counts[12]+counts[13]-counts[14]+counts[15]);
P_dc=2**-0.5*(-counts[8]+counts[9]+counts[10]-counts[11]+counts[12]-counts[13]-counts[14]+counts[15]);
CHSH=P_ab-P_ac+P_db+P_dc;

