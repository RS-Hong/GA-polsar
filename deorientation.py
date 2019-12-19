from readbin import readbin,savebin,savehdr
import numpy as np
import math
import os

wdirname = "C:\\Users\\Administrator\\Desktop\\oa_ESAR_Oberpfaffenhofen_LEE\\T3"
rdirname = "C:\\Users\\Administrator\\Desktop\\ESAR_Oberpfaffenhofen_LEE\\T3"

lines = 1408
samples = 1540

t11 = readbin(lines,samples,os.path.join(rdirname,"T11.bin"))
t22 = readbin(lines,samples,os.path.join(rdirname,"T22.bin"))
t33 = readbin(lines,samples,os.path.join(rdirname,"T33.bin"))
t12_real = readbin(lines,samples,os.path.join(rdirname,"T12_real.bin"))
t12_imag = readbin(lines,samples,os.path.join(rdirname,"T12_imag.bin"))
t13_real = readbin(lines,samples,os.path.join(rdirname,"T13_real.bin"))
t13_imag = readbin(lines,samples,os.path.join(rdirname,"T13_imag.bin"))
t23_real = readbin(lines,samples,os.path.join(rdirname,"T23_real.bin"))
t23_imag = readbin(lines,samples,os.path.join(rdirname,"T23_imag.bin"))


lines,samples = t11.shape
B = (t22-t33) / 2
E = t23_real
OA = np.arccos(B / (B**2 + E**2)**0.5)/4    #根据公式计算方位角矩阵
loc = np.where((E / (B**2 + E**2)**0.5) < 0)
OA[loc] = OA[loc] * -1

t12 = t12_real + 1j * t12_imag
t13 = t13_real + 1j * t13_imag
t23 = t23_real + 1j * t23_imag
t21 = np.conj(t12)
t31 = np.conj(t13)
t32 = np.conj(t23)

#共极化通道能量集中于HH
# oa2 = -2 * OA
# oa_t12 = t12 * np.cos(oa2) + t13 * np.sin(oa2)
# oa_t12_real = np.real(oa_t12)
# ero_oa = oa_t12_real < 0 #安文韬方法中集中于VV通道取向角位置
# OA[ero_oa & (OA < 0)] = OA[ero_oa & (OA < 0)] + math.pi/2
# OA[ero_oa & (OA > 0)] = OA[ero_oa & (OA > 0)] - math.pi/2


#计算旋转过后的T矩阵
oa2 = -2 * OA
oa_t12 = t12 * np.cos(oa2) + t13 * np.sin(oa2)
oa_t13 = -1 * t12 * np.sin(oa2) + t13 * np.cos(oa2)
oa_t22 = np.real(np.cos(oa2) * (t22 * np.cos(oa2) + t32 * np.sin(oa2)) + np.sin(oa2) * (t23 * np.cos(oa2) + t33 * np.sin(oa2)))
oa_t23 = np.cos(oa2) * (t23 * np.cos(oa2) + t33 * np.sin(oa2)) - np.sin(oa2) * (t22 * np.cos(oa2) + t32 * np.sin(oa2))
oa_t33 = np.real(-1 * np.sin(oa2) * (t32 * np.cos(oa2) - t22 * np.sin(oa2)) + np.cos(oa2) * (t33 * np.cos(oa2) - t23 * np.sin(oa2)))

#取出实部虚部
oa_t12_real = np.real(oa_t12)
oa_t12_imag = np.imag(oa_t12)
oa_t13_real = np.real(oa_t13)
oa_t13_imag = np.imag(oa_t13)
oa_t23_real = np.real(oa_t23)
oa_t23_imag = np.imag(oa_t23)


# 得到旋转后矩阵的路径
oa_T11 = os.path.join(wdirname,'T11.bin')
oa_T12_real = os.path.join(wdirname,'T12_real.bin')
oa_T12_imag = os.path.join(wdirname,'T12_imag.bin')
oa_T13_real = os.path.join(wdirname,'T13_real.bin')
oa_T13_imag = os.path.join(wdirname,'T13_imag.bin')
oa_T22 = os.path.join(wdirname,'T22.bin')
oa_T23_real = os.path.join(wdirname,'T23_real.bin')
oa_T23_imag = os.path.join(wdirname,'T23_imag.bin')
oa_T33 = os.path.join(wdirname,'T33.bin')
oa_path = os.path.join(wdirname,'OA.bin')
# 保存旋转后的T矩阵


savebin(oa_T11,t11)
savebin(oa_T12_real,oa_t12_real)
savebin(oa_T12_imag,oa_t12_imag)
savebin(oa_T13_real,oa_t13_real)
savebin(oa_T13_imag,oa_t13_imag)
savebin(oa_T22,oa_t22)
savebin(oa_T23_real,oa_t23_real)
savebin(oa_T23_imag,oa_t23_imag)
savebin(oa_T33,oa_t33)
savebin(oa_path,OA)

savehdr(wdirname,'T11',lines,samples)
savehdr(wdirname,'T12_real',lines,samples)
savehdr(wdirname,'T12_imag',lines,samples)
savehdr(wdirname,'T13_real',lines,samples)
savehdr(wdirname,'T13_imag',lines,samples)
savehdr(wdirname,'T22',lines,samples)
savehdr(wdirname,'T23_real',lines,samples)
savehdr(wdirname,'T23_imag',lines,samples)
savehdr(wdirname,'T33',lines,samples)
savehdr(wdirname,'OA',lines,samples)