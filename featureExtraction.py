import numpy as numpy
import scipy as sp
import sp_peak_amp_freq
import sp_pitch_period
import formants
import LPCC
import RCC
import lsf
import hjorth
import dwt
import sigproc

def extract(filtered_audio, Fs):
	nfft = 256
	num_bins = 40
	start_frequency = 150
	end_frequency = 3200
	c = filtered_audio.shape[0]
	features = []
	for i in range(c):
		framed_audio = sigproc.framesig(filtered_audio[i],256,128)
		features.append([])
		j = framed_audio.shape[0]

		if j > 10:
			req_frames = framed_audio[int(j/2)-5:int(j/2)+5]
			print(req_frames)
		for k in range(len(req_frames)):
			peak_amp, peak_freq = sp_peak_amp_freq.peakFreq(req_frames[k],50)
			pitch_periods = sp_pitch_period.pitch_period(req_frames[k],Fs)
			form = formants.formant(req_frames[k])
			cep = LPCC.lpcc(req_frames[k])
			real_cc = RCC.rcc(req_frames[k])
			lsfs = lsf.LSF(req_frames[k])
			hjorth_parameters = hjorth.params(req_frames[k])
			wavelet = dwt.wenergy(req_frames[k],'db7',5);
			features[i].extend(lsfs)
			features[i].extend(hjorth_parameters)
			features[i].extend(wavelet)
	return features
