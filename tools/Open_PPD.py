import pandas as pd
import os
import numpy as np
import pylab as plt
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize

plt.rcParams['figure.figsize'] = [14,12]
plt.rcParams['axes.xmargin'] = 0

from data_import import import_ppd

#data = import_ppd(r'C:\Users\vr_ephys\Desktop\pyPhotometry_v0.3\data\1318_HC-2021-11-08-152647.ppd', low_pass=20, high_pass=0.001)
#print(data)

#in data_filename replace filepath with datafile you want to analyse

data_folder = 'data'
data_filename = (r'/mnt/chromeos/MyFiles/Downloads/1318_HC-2021-11-12-151412.ppd')
data = import_ppd(os.path.join(data_folder, data_filename))

#If you have a CSV file, it will give you a JSON and CSV. Here is code to open the JSON file.
#import json
#with open(r'C:\Users\vr_ephys\Desktop\pyPhotometry_v0.3\data\TEST-2021-10-28-142416.json') as json_file:
 #   data1 = json.load(json_file)
  #  print(data1)

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))
print('......printing PyPhotometry data......')
print(df)

print('------------------------------------------------------------------------------\n .......plotting raw GCaMP and isosbestic signal......')
GCaMP_raw = data['analog_1']
isosbestic_raw = data['analog_2']
time_seconds = data['time']/1000
sampling_rate = data['sampling_rate']

plt.plot(time_seconds, GCaMP_raw, 'g', label='GCaMP')
plt.plot(time_seconds, isosbestic_raw, 'r', label='Isosbestic')
plt.xlabel('Time (seconds)')
plt.ylabel('Signal (volts)')
plt.title('Raw signals')
plt.legend()
plt.show()

#large electrical noise artefacts should be attenuated with median filter, then low-pass filter.
#zero-phase filter with a 10 Hz cut-off frequency
print('......plotting de-noised GCaMP and isosbestic signals......')

#medfilter to remove electrical artefacts
GCaMP_denoised = medfilt(GCaMP_raw, kernel_size=5)
isosbestic_denoised = medfilt(isosbestic_raw, kernel_size=5)

b, a = butter(2, 10, btype='low', fs = sampling_rate)
GCaMP_denoised = filtfilt(b,a, GCaMP_denoised)
isosbestic_denoised = filtfilt(b,a, isosbestic_denoised)

#plot denoised signals
plt.plot(time_seconds, GCaMP_denoised, 'g', label='GCaMP denoised')
plt.plot(time_seconds, isosbestic_denoised, 'r', label='isosbestic denoised')
plt.xlabel('Time (seconds)')
plt.ylabel('Signal (volts)')
plt.title('Denoised signals')
plt.legend
plt.show()

#zoom in on x-axis to see how lowpass filtered smoothens the signal
print('......plotting de-noised GCaMP and isosbestic signals over raw signal......')
plt.plot(time_seconds, GCaMP_raw, label='GCaMP raw')
plt.plot(time_seconds, isosbestic_raw, label='isosbestic raw')
plt.plot(time_seconds, GCaMP_denoised, label='GCaMP denoised')
plt.plot(time_seconds, isosbestic_denoised, label='isosbestic denoised')
plt.xlabel('Time (seconds)')
plt.ylabel('Signal (volts)')
plt.title('Denoised signals and raw signals (GCaMP & isosbestic)')
plt.legend()
plt.xlim(0,60)
plt.ylim(0.2,1.0)
plt.show()

#Correction for photobleaching
#remove slow changes > highpass filter at a 0.001 Hz cut off freq.
#removes drift due to bleaching, and any physiological variation in the signal on very slow timescales
print('\n......correcting for photobleaching......')


b,a = butter(2, 0.001, btype='high', fs=sampling_rate)
GCaMP_highpass = filtfilt(b,a, GCaMP_denoised, padtype='even')
isosbestic_highpass = filtfilt(b,a, isosbestic_denoised, padtype='even')

plt.plot(time_seconds, GCaMP_highpass    ,'g', label='GCaMP highpass')
plt.plot(time_seconds, isosbestic_highpass-0.1,'r', label='Isosbestic highpass')
plt.xlabel('Time (seconds)')
plt.ylabel('Signal (volts)')
plt.title('Bleaching correction by highpass filtering')
plt.legend()
plt.show()

#Another form of photobleaching correction is to subtrac the low order polynomial fit
#allows for more DoF in the slow component that is removed
"""coefs_GCaMP = np.polyfit(time_seconds, GCaMP_denoised, deg=4)
GCaMP_polyfit = np.polyval(coefs_GCaMP, time_seconds)

# Fit 4th order polynomial to isosbestic signal.
coefs_iso = np.polyfit(time_seconds, isosbestic_denoised, deg=4)
isosbestic_polyfit = np.polyval(coefs_iso, time_seconds)

# Plot fits
plt.plot(time_seconds, GCaMP_denoised, 'g', label='GCaMP')
plt.plot(time_seconds, GCaMP_polyfit,'k', linewidth=1.5)
plt.plot(time_seconds, isosbestic_denoised, 'r', label='Isosbestic')
plt.plot(time_seconds, isosbestic_polyfit,'k', linewidth=1.5)
plt.title('Polynomial fit to bleaching.')
plt.xlabel('Time (seconds)');

GCaMP_ps = GCaMP_denoised - GCaMP_polyfit
isosbestic_ps = isosbestic_denoised - isosbestic_polyfit

plt.plot(time_seconds, GCaMP_ps    , 'g', label='GCaMP')
plt.plot(time_seconds, isosbestic_ps-0.1, 'r', label='Isosbestic')
plt.title('Bleaching correction by subtraction of polynomial fit')
plt.xlabel('Time (seconds)');
plt.show()"""

#Motion correction > fin the best linear fit of the isosbestic signal to GCaMP
#subtract estimated motion component from GCaMP
print('......correcting for motion artefacts......')
slope, intercept, r_value, p_value, std_err = linregress(x=isosbestic_highpass, y=GCaMP_highpass)

plt.scatter(isosbestic_highpass[::5], GCaMP_highpass[::5],alpha=0.1, marker='.')
x = np.array(plt.xlim())
plt.plot(x, intercept+slope*x)
plt.xlabel('Isosbestic')
plt.ylabel('GCaMP')
plt.title('Isosbestic - GCaMP correlation.')
plt.show()
print('Slope    : {:.3f}'.format(slope))
print('R-squared: {:.3f}'.format(r_value**2))

GCaMP_est_motion = intercept + slope * isosbestic_highpass
GCaMP_corrected = GCaMP_highpass - GCaMP_est_motion

isosbestic_est_motion = intercept + slope * GCaMP_highpass
isosbestic_corrected = isosbestic_highpass - isosbestic_est_motion

plt.plot(time_seconds, GCaMP_highpass  , label='GCaMP - pre motion correction')
plt.plot(time_seconds, GCaMP_corrected, 'g', label='GCaMP - motion corrected')
plt.plot(time_seconds, GCaMP_est_motion - 0.05, 'y', label='estimated motion')
plt.xlabel('Time (seconds)')
plt.title('Motion correction')
plt.legend()
plt.xlim(0,180);
plt.show()

#Convert motion correction signal to delta F / F
#estimate F as a function of session time (photobleaching) by lowpass filtering

print('\n......converting to dF/F signal......')
b,a = butter(2, 0.001, btype='low', fs=sampling_rate)
baseline_fluorescence = filtfilt(b,a, GCaMP_denoised, padtype='even')

plt.plot(time_seconds, GCaMP_denoised       , 'g', label='GCaMP denoised')
plt.plot(time_seconds, baseline_fluorescence, 'k', label='baseline fluorescence')
plt.xlabel('Time (seconds)')
plt.title('Baseline fluorescence')
plt.legend();
GCaMP_dF_F = GCaMP_corrected/baseline_fluorescence
plt.show()

plt.plot(time_seconds, GCaMP_dF_F*100, 'g')
plt.xlabel('Time (seconds)')
plt.ylabel('GCaMP delta-F/F (%)')
plt.title('GCaMP dF/F')
plt.xlim(0,180);
plt.show()

b,a = butter(2, 0.001, btype='low', fs=sampling_rate)
baseline_fluorescence = filtfilt(b,a, isosbestic_denoised, padtype='even')

plt.plot(time_seconds, isosbestic_denoised       , 'g', label='Isosbestic denoised')
plt.plot(time_seconds, baseline_fluorescence, 'k', label='baseline fluorescence')
plt.xlabel('Time (seconds)')
plt.title('Baseline fluorescence')
plt.legend();
isosbestic_dF_F = isosbestic_corrected/baseline_fluorescence
plt.show()

plt.plot(time_seconds, isosbestic_dF_F*100, 'r')
plt.xlabel('Time (seconds)')
plt.ylabel('Isosbestic delta-F/F (%)')
plt.title('Isosbestic dF/F')
plt.xlim(0,180);
plt.show()