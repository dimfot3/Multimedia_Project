import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import wavfile
sys.path.insert(0, 'src')
import utils
from encoder0 import codec0, coder0, decoder0


# load the prototype response
h_pr = np.load('./protocol_files/h.npy', allow_pickle=True).tolist()['h'].reshape(-1, )
M = 32
N = 36

# 1.calculate the H and G
H = utils.make_mp3_analysisfb(h_pr, M)
G = utils.make_mp3_synthesisfb(h_pr, M)

# 2 and 3. ploting H in vs Hz and Bark in dB. I assume sampling rate 44100Hz
# f, ax = plt.subplots(1, 2)
# for i in range(M):
#     Hi = np.abs(np.fft.fft(H[:, i]))[0 : H.shape[0] // 2]
#     freqs = np.fft.fftfreq(H.shape[0], 1/(44100))[0 : H.shape[0] // 2]
#     ax[0].plot(freqs, 20*np.log10(Hi))
#     ax[0].set_ylim([-10, 1])
#     ax[0].set_ylabel('dB', fontsize=20)
#     ax[0].set_xlabel('Hz', fontsize=20)
#     ax[0].set_title('h filters response Hz', fontsize=20, fontweight='bold')
#     ax[0].tick_params(axis='x', which='both', labelsize=16)
#     ax[0].tick_params(axis='y', which='both', labelsize=16)
#     z = 13*np.arctan(0.00076*freqs) + 3.5*np.arctan((freqs/7500)**2)
#     ax[1].plot(z, 20*np.log10(Hi))
#     ax[1].set_ylim([-10, 1])
#     ax[1].set_ylabel('dB', fontsize=20)
#     ax[1].set_xlabel('Bark', fontsize=20)
#     ax[1].set_title('h filter response Bark', fontsize=20, fontweight='bold')
#     ax[1].tick_params(axis='x', which='both', labelsize=16)
#     ax[1].tick_params(axis='y', which='both', labelsize=16)
# plt.show()


music_file = './data/myfile.wav'
xhat, Y_tot = codec0(music_file, h_pr, M, N)
samplerate, data = wavfile.read(music_file)

plt.plot(data, label='x real')
plt.plot(xhat, label='$\hat{x}$')
plt.legend()
plt.show()
wavfile.write('test.wav', samplerate, xhat.astype('int16'))
# # for i in range(200):
# #     xtest = np.roll(xhat.astype('float32'), 100-i)
# #     print(np.corrcoef(data.astype('float32'), xtest)[0, 1], 100-i)

plt.plot(data.astype('float32') - xhat.astype('float32'))
plt.title('SNR', fontsize=20, fontweight='bold')
plt.xlabel('samples', fontsize=20)
plt.ylabel('SNR', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


Y_tot = coder0(music_file, h_pr, M, N)
xhat = decoder0(Y_tot, h_pr, M, N)
wavfile.write('test.wav', samplerate, xhat.astype('int16'))
plt.plot(xhat, label='$\hat{x}$')
plt.plot(data, label='x real')
plt.legend()
plt.show()




