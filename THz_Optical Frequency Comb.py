import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import fft
 
def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)
#mean为分布的平均值，亦即偏移量
#sigma为分布的半高宽

# 返回某区间内频率梳齿位置，ulim为频率上限，llim为频率下限，FreDom为频域空间坐标，FreInt为FFT之后的频域信号强度
def Freq_max(llim, ulim, FreDom, FreInt):
    pos_index = np.where((FreDom > llim) & (FreDom < ulim))
    pos_index = np.asarray(pos_index[0])  #将min_index max_index结构由tuple转换为array
    p_llim = pos_index[0]  #数据下限
    p_ulim = pos_index[np.size(pos_index)-1]  #数据上限
    comb_intensity = max(FreInt[p_llim: p_ulim])
    comb_pos = np.where(FreInt[p_llim: p_ulim] == comb_intensity)
    comb_pos = np.asarray(comb_pos[0])+p_llim
    return comb_pos, comb_intensity


#区间自动寻峰函数
#本函数基于微分方法实现，对信号做log处理之后再一阶微分
#基于微分幅值正负变化寻峰
#FreDom类比为x，FreInt类比为y，现要求y对x的导数
def Peak_seeking(llim, ulim, thres, FreDom, FreInt):
    x_index = np.where((FreDom > llim) & (FreDom < ulim))  #找到x区间
    x_index = np.asarray(x_index[0])
    x_llim = x_index[0]
    x_ulim = x_index[np.size(x_index)-1]
    s = np.size(x_index)
    f_diff = np.linspace(0, s, s)*0
    f_log = np.log(abs(FreInt[x_llim: x_ulim]))  #对区间数据取绝对值的log
    for i in range(1, s-1):
        f_diff[i] = f_log[x_llim+i]-f_log[x_llim+i-1]  #求导，因为数据间隔为1，故只做减法即可
    #此时f_diff size为s
    peak_num = 0
    peak_list = np.linspace(0, s, s)*0 #最多只有s个峰
    for i in range(1, s-1):
        if (f_diff[i]>0) & (f_diff[i+1]<0) & (f_diff[i]-f_diff[i+1]>1):
            peak_list[peak_num] = FreDom[x_llim + i]
            peak_num += 1
    return peak_list, peak_num, f_diff

n_pulse = 1000 #脉冲数量，实际过程应当为总采样时间/采样频率，
               #由于高斯分布不支持极小时域信号的输入，故将其延拓到[0, 1s]时域内展开
pdr = 0.01 #pulse duration ratio 脉空占宽比
f1 = 1e8 #激光重频 此处为100 MHz，也是光频梳的梳齿间隔
c = 3e8 #光速
lam = 1500 #光波长 1550 nm
f2 = 3e8*1e9/lam #光频
f3 = f2/80e6 #相对频率
f_r = 1e12 #采样频率，设定为1 THz
n_spp = int(f_r/f1) #每个周期内采样点数
n_sample = n_pulse*n_spp #总采样点数
t_all = n_pulse  #总采样时间
t_signal = pdr*n_spp #脉冲宽度
t_sample = np.linspace(0, n_sample, n_sample) 
sig_sample = np.linspace(0, n_sample, n_sample)*0 
phs_sample = np.linspace(0, n_sample, n_sample)*0  #相移，假设每个波包的相移为一固定值
mean = 0.5*t_signal
for i in range(1, n_pulse+1):
    x_i = np.linspace(0, t_signal, int(t_signal))
    sig_sample[(i-1)*n_spp: (i-1)*n_spp+int(t_signal)] = normal_distribution(x_i, mean, 0.1*t_signal)
    phs_sample[(i-1)*n_spp: (i-1)*n_spp+int(t_signal)] = i

y_light1 = np.sin(2*np.pi*0.1*t_sample)*sig_sample
y_light2 = np.sin(2*np.pi*0.1*t_sample+phs_sample*np.pi/2)*sig_sample
# for i in range(1, 11):    
#     #y_light += np.sin(2*np.pi*0.1*i*t_sample+phs_sample*np.pi)*sig_sample
#     y_light1 = y_light1 + np.sin(np.pi*0.1*i*t_sample)
    
#此处采用的光频率为0.1~0.9 Hz，映射到实际频率为0.1*100M*10000=1e11 Hz，即0.1~0.9 THz
#再考虑到载波的展宽（100MHz），实际的光频梳约有100根梳齿（0.095~1.005 THz，1e10）
Fre_Dom = np.linspace(-0.5, 0.5, n_sample)*n_spp*f1
Fre1 = fft.fft(y_light1)
Fre1 = fft.fftshift(Fre1)

Fre2 = fft.fft(y_light2)
Fre2 = fft.fftshift(Fre2)

#定位模拟出来的光频梳梳齿数值位置
comb_pos_sft, comb_intensity_sft = Freq_max(0.994e11, 1.006e11, Fre_Dom, Fre2)

# #时域图像
plt.figure(figsize=(16,6), dpi = 500)
plt.title('Enveloped Thz Pulses')
ax1 = plt.subplot(131)
plt.plot(t_sample, sig_sample)
plt.plot(t_sample, y_light1)
plt.xlim(0,100)
plt.grid()

ax2 = plt.subplot(132)
plt.plot(t_sample, sig_sample)
plt.plot(t_sample, y_light1)
plt.xlim(10000, 10100)
plt.grid()
ax = plt.gca()
ax.set_yticklabels([])

ax3 = plt.subplot(133)
plt.plot(t_sample, sig_sample)
plt.plot(t_sample, y_light1)
plt.xlim(20000, 20100)
plt.grid()
ax = plt.gca()
ax.set_yticklabels([])
plt.show()

#频域图像-对数坐标
plt.figure(figsize=(12,6), dpi = 400)
ax1 = plt.subplot(121)
plt.plot(Fre_Dom*n_spp*f1, np.log(abs(Fre1)))
plt.grid()
ax = plt.gca()
ax.set_ylabel('ln|F|', fontsize = 15)

ax2 = plt.subplot(122)
plt.plot(Fre_Dom[int(n_sample*1/2): int(n_sample*3/4)], np.log(abs(Fre1[int(n_sample*1/2): int(n_sample*3/4)])))
plt.xlim(9.95e10, 10.05e10)
plt.grid()
ax = plt.gca()
ax.set_ylim(-3, 7)
plt.suptitle('FFT of THz Pulses in Log', fontsize = 20)
plt.show()

#频域图像-普通坐标
plt.figure(figsize=(12,6), dpi = 400)
ax1 = plt.subplot(121)
plt.plot(Fre_Dom, abs(Fre1))
plt.xlim(9e10, 11e10)
plt.grid()
ax = plt.gca()
ax.set_ylabel('|F|', fontsize = 15)
ax.set_ylim(0, 800)

ax2 = plt.subplot(122)
plt.plot(Fre_Dom*n_spp*f1, abs(Fre1))
plt.xlim(9.95e10, 10.05e10)
plt.grid()
ax = plt.gca()
ax.set_yticklabels([])
plt.suptitle('FFT of THz Pulses in Ordinary', fontsize = 20)
plt.show()

#考虑相移前后频域空间的对比
plt.figure(figsize = (12,9), dpi = 400)
plt.plot(Fre_Dom[int(comb_pos_sft-800):int(comb_pos_sft+800)], \
          abs(Fre1[int(comb_pos_sft-800):int(comb_pos_sft+800)]), \
              'Orange', linewidth = 2, label = 'Initial comb')
plt.plot(Fre_Dom[int(comb_pos_sft-800):int(comb_pos_sft+800)], \
          abs(Fre2[int(comb_pos_sft-800):int(comb_pos_sft+800)]), \
              'Green', linewidth = 2, label = 'Shifted Comb')
plt.grid()
plt.legend(fontsize = 20)
plt.text(1.0055e11, 300,abs(Fre_Dom[int(comb_pos_sft)]),color='Green',ha='center', fontsize = 20)  
ax = plt.gca()
ax.set_xticks((1.0045e11, 1.0050e11, 1.0055e11, 1.0060e11))
plt.show()

# #待测太赫兹信号
y_test1 = np.sin(2*np.pi*0.0901*t_sample) #没有相移
y_test2 = np.sin(2*np.pi*0.0901*t_sample+phs_sample*np.pi)
#假设有一频率为0.0901 THz的待测信号
y_beat1 = y_light1*y_test1 #拍频
y_beat2 = y_light2*y_test1 #
Fre_beat1 = fft.fft(y_beat1)
Fre_b1 = fft.fftshift(Fre_beat1)
Fre_beat2 = fft.fft(y_beat2)
Fre_b2 = fft.fftshift(Fre_beat2)

#拍频后的时域图像-1
plt.figure(figsize=(4,3), dpi = 400)
plt.plot(t_sample[0: 10000], y_light1[0: 10000], 'orange', label = 'initial')
plt.plot(t_sample[0: 10000], y_beat1[0: 10000], 'green', label = 'beated')
plt.xlim(0,100)
plt.grid()
ax = plt.gca()
plt.title('Beated THz Pulses in Time Domain-1', fontsize = 15)
plt.legend()
plt.show()

# #拍频后的频域图像-1
# plt.figure(figsize=(18,6), dpi = 400)
# ax1 = plt.subplot(131)
# plt.plot(Fre_Dom[int(n_sample*1/4): int(n_sample*3/4)], \
#           abs(Fre_b1[int(n_sample*1/4): int(n_sample*3/4)]))
# plt.grid()
# ax = plt.gca()
# ax.set_ylabel('|F|')
# ax.set_xlim(-2e11, 2e11)

# ax1 = plt.subplot(132)
# plt.plot(Fre_Dom[int(n_sample*9/20): int(n_sample*11/20)], \
#           np.log(abs(Fre_b1[int(n_sample*9/20): int(n_sample*11/20)])))
# plt.grid()
# ax = plt.gca()
# ax.set_ylabel('ln|F|')
# ax.set_xlim(0, 0.5e11)
# #ax.set_ylim(-18, -8)

# ax1 = plt.subplot(133)
# plt.plot(Fre_Dom[int(n_sample*19/40): int(n_sample*21/40)], \
#           abs(Fre_b1[int(n_sample*19/40): int(n_sample*21/40)]))
# plt.grid()
# ax = plt.gca()
# ax.set_xlim(0, 1e9)
# #ax.set_ylim(0, 3e-4)
# ax.set_yticklabels([])
# plt.suptitle('FFT of Beated THz Pulses-1', fontsize = 20)
# plt.show()

# #拍频后的时域图像-2
# plt.figure(figsize=(4,3), dpi = 400)
# plt.plot(t_sample[0: 10000], y_light2[0: 10000], 'orange', label = 'initial')
# plt.plot(t_sample[0: 10000], y_beat2[0: 10000], 'green', label = 'beated')
# plt.xlim(0,100)
# plt.grid()
# ax = plt.gca()
# plt.title('Beated THz Pulses in Time Domain-2', fontsize = 15)
# plt.legend()
# plt.show()

#拍频后的频域图像-2
# plt.figure(figsize=(18,6), dpi = 400)
# ax1 = plt.subplot(131)
# plt.plot(Fre_Dom[int(n_sample*1/4): int(n_sample*3/4)], \
#          abs(Fre_b2[int(n_sample*1/4): int(n_sample*3/4)]))
# plt.grid()
# ax = plt.gca()
# ax.set_ylabel('ln|F|')
# ax.set_xlim(-2e11, 2e11)

# ax1 = plt.subplot(132)
# plt.plot(Fre_Dom[int(n_sample*3/8): int(n_sample*5/8)], \
#          np.log(abs(Fre_b2[int(n_sample*3/8): int(n_sample*5/8)])))
# plt.grid()
# ax = plt.gca()
# ax.set_ylabel('|F|')
# ax.set_xlim(-0.5e11, 0.5e11)
# #ax.set_ylim(-18, -8)

# ax1 = plt.subplot(133)
# plt.plot(Fre_Dom[int(n_sample*3/8): int(n_sample*5/8)], \
#          abs(Fre_b2[int(n_sample*3/8): int(n_sample*5/8)]))
# plt.grid()
# ax = plt.gca()
# ax.set_xlim(0, 1e9)
# #ax.set_ylim(0, 3e-4)
# ax.set_yticklabels([])
# plt.suptitle('FFT of Beated THz Pulses-2', fontsize = 20)
# plt.show()

# 两类拍频信号的频率差
comb_pos1, comb_intensity1 = Freq_max(0.05e9, 0.15e9, Fre_Dom, Fre1)
comb_pos2, comb_intensity2 = Freq_max(-0.05e9, 0.05e9, Fre_Dom, Fre2)

# 两类拍频之间的频域信号对比
plt.figure(figsize=(12,9), dpi = 400)
plt.plot(Fre_Dom[int(n_sample*1/2): int(n_sample*11/20)], \
         abs(Fre_b1[int(n_sample*1/2): int(n_sample*11/20)]),\
             'Orange', linewidth = 2, label = 'Initial')
plt.plot(Fre_Dom[int(n_sample*1/2): int(n_sample*11/20)], \
         abs(Fre_b2[int(n_sample*1/2): int(n_sample*11/20)]),\
             'Green', linewidth = 2, label = 'Shifted')
plt.grid()
plt.legend(fontsize = 20)
plt.title('FFT Comparison of Beated THz Pulses', fontsize = 25)
ax = plt.gca()
ax.set_xlim(0, 1e9)
#ax.set_ylim(0, 1.5e-4)
#ax.set_yticklabels([0, 4e-5, 8e-5, 12e-5, 16e-5, 20e-5])
plt.text(0.15e9, 220, abs(Fre_Dom[int(comb_pos2)]),color='Green',ha='center', fontsize = 25)  
plt.text(0.15e9, 400, abs(Fre_Dom[int(comb_pos1)]),color='Orange',ha='center', fontsize = 25)  
plt.show()

# #
# plt.figure(figsize=(18,6), dpi = 500)
# ax1 = plt.subplot(121)
# plt.plot(Fre_Dom, abs(Fre_b1))
# plt.grid()
# ax = plt.gca()
# ax.set_xlim(1.95e11, 2.05e11)
# ax.set_ylabel('|F|')

# ax1 = plt.subplot(122)
# plt.plot(Fre_Dom, abs(Fre_b1))
# plt.grid()
# ax = plt.gca()
# ax.set_xlim(1.995e11, 2.005e11)
# ax.set_yticklabels([])
# plt.suptitle('FFT of Beated THz Pulses(2)', fontsize = 20)
# plt.show()

# 改变激光器重频，即频率梳梳齿间隔
f1_new = 1.01e8 #此处为101 MHz
Fre_Dom_new = np.linspace(-0.5, 0.5, n_sample)*n_spp*f1_new

# 两类拍频之间的频域信号对比
plt.figure(figsize=(12,9), dpi = 400)
plt.plot(Fre_Dom_new[int(n_sample*1/2): int(n_sample*11/20)], \
         abs(Fre_b1[int(n_sample*1/2): int(n_sample*11/20)]),\
             'Orange', linewidth = 2, label = 'Initial')
plt.plot(Fre_Dom_new[int(n_sample*1/2): int(n_sample*11/20)], \
         abs(Fre_b2[int(n_sample*1/2): int(n_sample*11/20)]),\
             'Green', linewidth = 2, label = 'Shifted')
plt.grid()
plt.legend(fontsize = 20)
plt.title('FFT Comparison of Beated THz Pulses', fontsize = 25)
ax = plt.gca()
ax.set_xlim(0, 1e9)
plt.text(0.15e9, 220, abs(Fre_Dom_new[int(comb_pos2)]),color='Green',ha='center', fontsize = 25)  
plt.text(0.15e9, 400, abs(Fre_Dom_new[int(comb_pos1)]),color='Orange',ha='center', fontsize = 25)  
plt.show()