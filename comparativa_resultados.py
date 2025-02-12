#%% Comparativa resultados NPs
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet
import re
from glob import glob
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy
from datetime import datetime,timedelta
import matplotlib as mpl
from scipy.interpolate import CubicSpline,PchipInterpolator
#%% funciones LECTOR RESULTADOS y  LECTOR CICLOS , calcular Hc
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                else:
                    # Capturar los casos con nombres de archivo en las últimas dos líneas
                    match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                    if match_files:
                        key = match_files.group(1)[2:]  # Obtener el nombre de la clave sin '# '
                        value = match_files.group(2)     # Obtener el nombre del archivo
                        meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=14,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

def calcular_hc(H, m):
    # Encuentra los índices donde m cruza el eje x (cambio de signo)
    # el error es la diferencia entre valor negativo y positivo
    cruces = np.where(np.diff(np.sign(m)) != 0)[0]

    hc_valores = []
    for i in cruces:
        # Interpolación lineal para encontrar el cruce exacto
        h1, h2 = H[i], H[i + 1]
        m1, m2 = m[i], m[i + 1]
        h_c = h1 - m1 * (h2 - h1) / (m2 - m1)
        hc_valores.append(h_c)
    
    # Obtén valores positivos y negativos
    hc_positivos = [h for h in hc_valores if h > 0]
    hc_negativos = [h for h in hc_valores if h < 0]

    # Calcula el promedio absoluto de los positivos y negativos
    if hc_positivos and hc_negativos:
        hc_promedio = (np.mean(hc_positivos) + abs(np.mean(hc_negativos))) / 2
        hc_err=    hc_valores[0]+hc_valores[1] 
    else:
        hc_promedio = None  # Si no hay suficientes cruces
    print('Hc = ', hc_promedio,hc_valores)
    return hc_promedio, hc_err


#%% Ciclos promedio 
dir_2024=os.path.join('..','250206_NF241126','100kHz_7_to_10_2024') 
ciclos_2024 = glob(os.path.join(dir_2024,'**', '*ciclo_promedio*'),recursive=True)
ciclos_2024.sort()
labels_2024 = ['N1_108_'+os.path.split(s)[-1].split('_')[1].split('dA')[0] for s in ciclos_2024]
print(f'Cargados {len(ciclos_2024)} ciclos del directorio {dir_2024}')

dir_2025_A=os.path.join('..','250206_NF241126','108kHz_2_to_15_2025')
ciclos_2025_A = glob(os.path.join(dir_2025_A,'**', '*ciclo_promedio*'),recursive=True)
ciclos_2025_A.sort()
labels_2025_A = ['N1_108_'+os.path.split(s)[-1].split('_')[1].split('dA')[0] for s in ciclos_2025_A]
print(f'Cargados {len(ciclos_2025_A)} ciclos del directorio {dir_2025_A}')

dir_2025_B=os.path.join('108kHz_2_to_15')
ciclos_2025_B = glob(os.path.join(dir_2025_B,'**', '*ciclo_promedio*'),recursive=True)
ciclos_2025_B.sort()
labels_2025_B = ['N1_108_'+os.path.split(s)[-1].split('_')[1].split('dA')[0] for s in ciclos_2025_B]
print(f'Cargados {len(ciclos_2025_B)} ciclos del directorio {dir_2025_B}')

dir_2025_C=os.path.join('..','250212_NF250211_repeticion','108kHz_2_to_8')
ciclos_2025_C = glob(os.path.join(dir_2025_C,'**', '*ciclo_promedio*'),recursive=True)
ciclos_2025_C.sort()
labels_2025_C = ['N1_108_'+os.path.split(s)[-1].split('_')[1].split('dA')[0] for s in ciclos_2025_C]
print(f'Cargados {len(ciclos_2025_C)} ciclos del directorio {dir_2025_C}')

dir_2025_D=os.path.join('..','250212_NF250211_repeticion','108kHz_8_to_15')
ciclos_2025_D = glob(os.path.join(dir_2025_D,'**', '*ciclo_promedio*'),recursive=True)
ciclos_2025_D.sort()
labels_2025_D = ['N1_108_'+os.path.split(s)[-1].split('_')[1].split('dA')[0] for s in ciclos_2025_D]
print(f'Cargados {len(ciclos_2025_D)} ciclos del directorio {dir_2025_D}')

#%% Levanto archivos resultados
res_2024 = glob(os.path.join(dir_2024,'**','*resultados*'),recursive=True)
res_2024.sort()
print(f'Cargados {len(res_2024)} resultados del directorio {dir_2024}')

res_2025_A = glob(os.path.join(dir_2025_A,'**','*resultados*'),recursive=True)
res_2025_A.sort()
print(f'Cargados {len(res_2025_A)} resultados del directorio {dir_2025_A}')

res_2025_B = glob(os.path.join(dir_2025_B,'**','*resultados*'),recursive=True)
res_2025_B.sort()
print(f'Cargados {len(res_2025_B)} resultados del directorio {dir_2025_B}')

res_2025_C = glob(os.path.join(dir_2025_C,'**','*resultados*'),recursive=True)
res_2025_C.sort()
print(f'Cargados {len(res_2025_C)} resultados del directorio {dir_2025_C}')

res_2025_D = glob(os.path.join(dir_2025_D,'**','*resultados*'),recursive=True)
res_2025_D.sort()
print(f'Cargados {len(res_2025_D)} resultados del directorio {dir_2025_D}')

#%%concateno C y D

ciclos_2025_C=ciclos_2025_C+ciclos_2025_D
labels_2025_C=labels_2025_C+ labels_2025_D
res_2025_C=res_2025_C+res_2025_D


#%% Ploteo ciclos
fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(nrows=2,ncols=2,figsize=(12,10), constrained_layout=True,sharey=True)



for i,p in enumerate(ciclos_2024):
    _,_,_,H,M,_=lector_ciclos(p)
    ax0.plot(H,M,label=labels_2024[i])

for i,p in enumerate(ciclos_2025_A):
    _,_,_,H,M,_=lector_ciclos(p)
    ax1.plot(H,M,label=labels_2025_A[i])

for i,p in enumerate(ciclos_2025_B):
    _,_,_,H,M,_=lector_ciclos(p)
    ax2.plot(H,M,label=labels_2025_B[i])
    
for i,p in enumerate(ciclos_2025_C):
    _,_,_,H,M,_=lector_ciclos(p)
    ax3.plot(H,M,label=labels_2025_C[i])

    
for a in [ax0,ax1,ax2]:    
    a.set_ylabel('M (A/m)')
    a.grid()
    #a.legend(ncol=1)
    a.set_xlabel('H (A/m)')

# ax0.set_title('2024')
# ax1.set_title('2025')
# ax2.set_title('2025')
# ax1.set_title('2025')
plt.suptitle('NF241126 @Citrato - N1 - 108kHz')
plt.savefig('Comparativa_NF_Citrato_108kHz', dpi=200, facecolor='w')
plt.show()   


#%% resultados SAR Tau Hc
(SAR_2024,SAR_2024_err,tau_2024,tau_2024_err,Hc_2024,Hc_2024_err,H_max_2024)=([],[],[],[],[],[],[])
for path in res_2024:
    _,_,_,_,_,Hc,campo_max,_,_,_,_,_,SAR,tau, _= lector_resultados(path)
    SAR_2024.append(np.mean(SAR))
    SAR_2024_err.append(np.std(SAR))
    tau_2024.append(np.mean(tau))
    tau_2024_err.append(np.std(tau))
    Hc_2024.append(np.mean(Hc))
    Hc_2024_err.append(np.std(Hc))
    H_max_2024.append(np.mean(campo_max)/1000)

(SAR_2025_A,SAR_2025_A_err,tau_2025_A,tau_2025_A_err,Hc_2025_A,Hc_2025_A_err,H_max_2025_A)=([],[],[],[],[],[],[])
for path in res_2025_A:
    _,_,_,_,_,Hc,campo_max,_,_,_,_,_,SAR,tau, _= lector_resultados(path)
    SAR_2025_A.append(np.mean(SAR))
    SAR_2025_A_err.append(np.std(SAR))
    tau_2025_A.append(np.mean(tau))
    tau_2025_A_err.append(np.std(tau))
    Hc_2025_A.append(np.mean(Hc))
    Hc_2025_A_err.append(np.std(Hc))
    H_max_2025_A.append(np.mean(campo_max)/1000)

(SAR_2025_B,SAR_2025_B_err,tau_2025_B,tau_2025_B_err,Hc_2025_B,Hc_2025_B_err,H_max_2025_B)=([],[],[],[],[],[],[])
for path in res_2025_B:
    _,_,_,_,_,Hc,campo_max,_,_,_,_,_,SAR,tau, _= lector_resultados(path)
    SAR_2025_B.append(np.mean(SAR))
    SAR_2025_B_err.append(np.std(SAR))
    tau_2025_B.append(np.mean(tau))
    tau_2025_B_err.append(np.std(tau))
    Hc_2025_B.append(np.mean(Hc))
    Hc_2025_B_err.append(np.std(Hc))
    H_max_2025_B.append(np.mean(campo_max)/1000)
 
 
(SAR_2025_C,SAR_2025_C_err,tau_2025_C,tau_2025_C_err,Hc_2025_C,Hc_2025_C_err,H_max_2025_C)=([],[],[],[],[],[],[])
for path in res_2025_C:
    _,_,_,_,_,Hc,campo_max,_,_,_,_,_,SAR,tau, _= lector_resultados(path)
    SAR_2025_C.append(np.mean(SAR))
    SAR_2025_C_err.append(np.std(SAR))
    tau_2025_C.append(np.mean(tau))
    tau_2025_C_err.append(np.std(tau))
    Hc_2025_C.append(np.mean(Hc))
    Hc_2025_C_err.append(np.std(Hc))
    H_max_2025_C.append(np.mean(campo_max)/1000) 
    
    
#%% SAR vs Hmax

fig,ax=plt.subplots(ncols=1,figsize=(7,5),sharey=True,constrained_layout=True)

ax.errorbar(x=H_max_2024,y=SAR_2024,yerr=SAR_2024_err,capsize=5,fmt='.-',label='2024')
ax.errorbar(x=H_max_2025_A,y=SAR_2025_A,yerr=SAR_2025_A_err,capsize=5,fmt='.-',label='2025 A')
ax.errorbar(x=H_max_2025_B,y=SAR_2025_B,yerr=SAR_2025_B_err,capsize=5,fmt='.-',label='2025 B')
ax.errorbar(x=H_max_2025_C,y=SAR_2025_C,yerr=SAR_2025_C_err,capsize=5,fmt='.-',label='2025 C')

ax.set_title('108 kHz')
ax.set_ylabel('SAR (W/g)')    
    
ax.grid()
ax.legend()
ax.set_xlabel('H$_{max}$ (kA/m)')
plt.suptitle('SAR vs H$_{max}$',fontsize=14)    
plt.savefig('SAR_vs_Hmax_108kHz.png',dpi=300)

#%% Tau vs Hmax

fig2,ax=plt.subplots(ncols=1,figsize=(7,5),sharey=True,constrained_layout=True)

ax.errorbar(x=H_max_2024,y=tau_2024,yerr=tau_2024_err,capsize=5,fmt='.-',label='2024')
ax.errorbar(x=H_max_2025_A,y=tau_2025_A,yerr=tau_2025_A_err,capsize=5,fmt='.-',label='2025 A')
ax.errorbar(x=H_max_2025_B,y=tau_2025_B,yerr=tau_2025_B_err,capsize=5,fmt='.-',label='2025 B')
ax.errorbar(x=H_max_2025_C,y=tau_2025_C,yerr=tau_2025_C_err,capsize=5,fmt='.-',label='2025 C')
ax.set_title('108 kHz')
ax.set_ylabel(r'$\tau$ (ns)')    
    
ax.grid()
ax.legend()
ax.set_xlabel('H$_{max}$ (kA/m)')
plt.suptitle(r'$\tau$ vs H$_{max}$',fontsize=14)    
plt.savefig('tau_vs_Hmax_108kHz.png',dpi=300)
#%% Hc vs Hmax

fig2,ax=plt.subplots(ncols=1,figsize=(7,5),sharey=True,constrained_layout=True)
ax.errorbar(x=H_max_2024,y=Hc_2024,yerr=Hc_2024_err,capsize=5,fmt='.-',label='2024')
ax.errorbar(x=H_max_2025_A,y=Hc_2025_A,yerr=Hc_2025_A_err,capsize=5,fmt='.-',label='2025 A')
ax.errorbar(x=H_max_2025_B,y=Hc_2025_B,yerr=Hc_2025_B_err,capsize=5,fmt='.-',label='2025 B')
ax.errorbar(x=H_max_2025_C,y=Hc_2025_C,yerr=Hc_2025_C_err,capsize=5,fmt='.-',label='2025 C')

ax.set_title('100 kHz')
ax.set_ylabel('H$_c$ (kA/m)')    
    
ax.grid()
ax.legend()
ax.set_xlabel('H$_{max}$ (kA/m)')
plt.suptitle('Coercitivo vs H$_{max}$',fontsize=14)   
plt.savefig('Hc_vs_Hmax_100kHz.png',dpi=300)


#%%
