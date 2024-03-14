import copy
from latqcdtools.statistics import bootstr
import numpy as np
import sys
#import math
import os
import subprocess

def get_mean(data):
    return np.mean(data[0], axis=0)


PathInputFolder="/u/shu1/test/TwoPtOut"
PathOutputFolder="/u/shu1/test/TwoPtOut"
os.chdir(PathInputFolder)
conflist=subprocess.check_output("ls | grep BulkCorr | awk -F _conf '{print $2}'", shell=True).decode("utf-8").split("\n")[:-1]
Nconf=len(conflist)
Nf=45
Nrsq=47
Nts=33

Ns=64

def compute_EE_mean(EE_data, Ns, factor):
    mean_sum_EE_full = np.mean(EE_data[0], axis=0)
    mean_EE_disc = np.mean(EE_data[1], axis=0)
    mean_sum_EE = mean_sum_EE_full - mean_EE_disc**2.*Ns**3.*factor
    return mean_sum_EE

ShearData, BulkData, Vacuum = [],[],[]
for conf in conflist:
    ShearData.append(np.loadtxt(PathInputFolder+"/ShearCorr_conf"+str(conf),dtype='f8'))
    BulkData.append(np.loadtxt(PathInputFolder+"/BulkCorr_conf"+str(conf),dtype='f8'))
    Vacuum.append(np.loadtxt(PathInputFolder+"/OnePt_conf"+str(conf),dtype='f8'))
ShearData=np.array(ShearData).reshape(Nconf, Nf, Nts, Nrsq, 4)
BulkData=np.array(BulkData).reshape(Nconf, Nf, Nts, Nrsq, 4)
Vacuum=np.array(Vacuum).reshape(Nconf, Nf, 12)
flowtimes=ShearData[0,:,0,0,0]
Rsq=ShearData[0,0,0,:,2]
dts=ShearData[0,0,:,0,1]

blocksize = 1
nblock_xyz = int(Ns/blocksize)
r_degen = [0] * int(3*(nblock_xyz/2+1)**2)
for x in range(nblock_xyz):
    dx = nblock_xyz-x if x > nblock_xyz/2 else x
    for y in range(nblock_xyz):
        dy = nblock_xyz-y if y > nblock_xyz/2 else y
        for z in range(nblock_xyz):
            dz = nblock_xyz-z if z > nblock_xyz/2 else z
            R = dx*dx + dy*dy +dz*dz
            r_degen[R] += 1
r_degen=np.array(r_degen)
R_sum = 0
for i in range(len(r_degen)):
    R_sum += r_degen[i]

r_sum_degen=[]
for ii in range(len(Rsq)):
    rsq=Rsq[ii]
    r_sum_degen.append(sum(r_degen[:int(rsq+1)]))
r_sum_degen=np.array(r_sum_degen)
factor = r_sum_degen/R_sum

ShearMeanErr, BulkMeanErr, VacuumMeanErr, BulkSubMeanErr = [],[],[],[]
for ift in range(Nf):
    sample_onept, mean_onept, error_onept = bootstr.bootstr(func=get_mean, data=[Vacuum[:,:,1:]], numb_samples=100, sample_size=len(Vacuum[:,:,1:]), same_rand_for_obs = False, seed=0, conf_axis=1, return_sample = True, nproc=2)
    tmp=[flowtimes[ift]]
    for iob in range(11):
        tmp.append(mean_onept[ift, iob])
        tmp.append(error_onept[ift, iob])
    VacuumMeanErr.append(tmp)
    for dt in range(Nts):
        Trace = np.swapaxes([Vacuum[:, ift, 1]]*Nrsq, 0, 1)
        EE_data=[BulkData[:, ift, dt, :, 3], Trace/Ns**3.]
        EE_sample, EE_mean, EE_err = bootstr.bootstr(func=compute_EE_mean, data=EE_data, numb_samples=100, sample_size=len(Trace), same_rand_for_obs = True, conf_axis=1, return_sample = True, nproc=8, args = {'Ns': Ns, 'factor': factor})
        for ir in range(Nrsq):
            BulkSubMeanErr.append([flowtimes[ift], dts[dt], Rsq[ir], EE_mean[ir], EE_err[ir]])
        shear=ShearData[:, ift, dt, :, 3]
        bulk=BulkData[:, ift, dt, :, 3]
        print("i_ft: ",ift, " dt: ", dt)
        sample_corr, mean_corr, error_corr = bootstr.bootstr(func=get_mean, data=[shear], numb_samples=100, sample_size=len(shear), same_rand_for_obs = False, seed=0, conf_axis=1, return_sample = True, nproc=2)
        for ir in range(Nrsq):
            ShearMeanErr.append([flowtimes[ift], dts[dt], Rsq[ir], mean_corr[ir], error_corr[ir]])
        sample_corr, mean_corr, error_corr = bootstr.bootstr(func=get_mean, data=[bulk], numb_samples=100, sample_size=len(bulk), same_rand_for_obs = False, seed=0, conf_axis=1, return_sample = True, nproc=8)
        for ir in range(Nrsq):
            BulkMeanErr.append([flowtimes[ift], dts[dt], Rsq[ir], mean_corr[ir], error_corr[ir]])
np.savetxt(PathOutputFolder+"/ShearMeanErr.dat", ShearMeanErr, fmt='%10.9e')
np.savetxt(PathOutputFolder+"/BulkMeanErr.dat", BulkMeanErr, fmt='%10.9e')
np.savetxt(PathOutputFolder+"/BulkSubMeanErr.dat", BulkSubMeanErr, fmt='%10.9e')
np.savetxt(PathOutputFolder+"/VacuumMeanErr.dat", VacuumMeanErr, fmt='%10.9e')
