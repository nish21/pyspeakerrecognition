import sigproc
import numpy as np
import scipy.io as sio
import sounddevice as sd

seed = sio.loadmat('../seed1250.mat')

def spec_sub(signal):
	NFFT = 1024
	frames = sigproc.framesig(signal, 256, 128)
	print(frames.shape)
	cspec = np.fft.fft(frames, NFFT)
	pspec = abs(cspec)
	print(pspec.shape)
	pspec *= pspec
	phase = np.angle(cspec)

	noise_est = np.mean(pspec[40:50])
	print(noise_est)
	clean_spec = pspec - noise_est
	print("1")
	#print (clean_spec)
	clean_spec[clean_spec < 0] = 0;
	print("2")
	#print(clean_spec)
	clean_spec **= 0.5
	clean_spec *= np.exp(phase)
	print("3")
	#print(clean_spec)
	reconstructed_frames = np.fft.ifft(clean_spec, NFFT)
	reconstructed_frames = np.real(reconstructed_frames)
	print(reconstructed_frames.shape)
	reconstructed_frames = reconstructed_frames[0:reconstructed_frames.shape[0],0:256]
	print(reconstructed_frames.shape)

	#print(reconstructed_frames.shape)
	#print(reconstructed_frames)
	enhanced_signal = sigproc.deframesig(reconstructed_frames, len(signal),256,128)
	#print(enhanced_signal)
	return enhanced_signal
#e_s = np.zeros((1,33075))
"""e_s = spec_sub(seed['audio'][300])
sio.savemat('saved.mat',{'vect':e_s})
print(e_s)
sd.default.samplerate = 11025
print("speak")
b = sd.rec(int(2*11025),samplerate=11025,channels = 1, blocking=True)
sd.play(seed['audio'][0], 11025)
"""
