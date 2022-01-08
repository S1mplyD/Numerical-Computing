import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics, io
from scipy import signal
from scipy.optimize import minimize
from numpy import fft

# Create a Gaussian kernel of size kernlen and standard deviation sigma
def gaussian_kernel(kernlen, sigma):
    x = np.linspace(- (kernlen // 2), kernlen // 2, kernlen)    
    # Unidimensional Gaussian kernel
    kern1d = np.exp(- 0.5 * (x**2 / sigma))
    # Bidimensional Gaussian kernel
    kern2d = np.outer(kern1d, kern1d)
    # Normalization
    return kern2d / kern2d.sum()

# Compute the FFT of the kernel 'K' of size 'd' padding with the zeros necessary
# to match the size of 'shape'
def psf_fft(K, d, shape):
    # Zero padding
    K_p = np.zeros(shape)
    K_p[:d, :d] = K

    # Shift
    p = d // 2
    K_pr = np.roll(np.roll(K_p, -p, 0), -p, 1)

    # Compute FFT
    K_otf = fft.fft2(K_pr)
    return K_otf

# Multiplication by A
def A(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(K * x))

# Multiplication by A transpose
def AT(x, K):
  x = fft.fft2(x)
  return np.real(fft.ifft2(np.conj(K) * x))

#Punto 1: Blur dell'immagine

#Import dell'immagine
image = io.imread('/content/imagecalc8.png')  

#Definisco il tipo dell'immagine e normalizzo l'immagine
X = image.astype(np.float64) / 255.0
m, n = X.shape

# Genera i diversi filtri di blur
K = psf_fft(gaussian_kernel(5, 0.5), 5, X.shape)
K1 = psf_fft(gaussian_kernel(7, 1), 7, X.shape)
K2 = psf_fft(gaussian_kernel(9, 1.3), 9, X.shape)

# Genera il rumore
dev = 0.033660826503758685 # np.random.uniform(low=0.0, high=0.05)
noise = np.random.normal(size=X.shape) * dev

# Aggiungi i vari livelli di blur e il rumore, per generare diverse immagini corrotte
b = A(X, K) + noise
b1 = A(X, K1) + noise
b2 = A(X, K2) + noise

#Calcolo del PSNR
PSNR = metrics.peak_signal_noise_ratio(X, b)
PSNR1 = metrics.peak_signal_noise_ratio(X, b1)
PSNR2 = metrics.peak_signal_noise_ratio(X, b2)

#Calcolo MSE
MSE = metrics.mean_squared_error(X, b)
MSE1 = metrics.mean_squared_error(X, b1)
MSE2 = metrics.mean_squared_error(X, b2)

#Punto 2: Soluzione naive

#Funzione che è necessario minimizzare per ottenere il risultato con soluzione naive
def f(x):
  J = x.reshape(m,n)
  res = 0.5*(np.linalg.norm(A(J, K)-b))**2
  return np.sum(res)

#Funzione per il calcolo del gradiente
def df(x):
  J = x.reshape(m,n)
  res = AT(A(J, K)-b, K)
  RES = np.reshape(res, m*n)
  return RES

result = minimize(f,X,method='CG',jac=df)

"""
Dal risultato della funzione minimize pongo in X1 un array contenente la matrice minimizzata,
ne eseguo il reshape per dargli la dimensione (512,512).
Normalizzo poi l'immagine
"""
Xnaive = np.array([result.x])
Xnaive = Xnaive.reshape(m,n)
X2naive = Xnaive.astype(np.float64) / 255

#Calcolo PSNR dell'immagine a cui è stato applicato il deblur con soluzione naive
b3 = A(X2naive,K) + noise
PSNR3 = metrics.peak_signal_noise_ratio(X2naive,b3)
MSE3 = metrics.mean_squared_error(X2naive, b3)
b31 = A(X2naive,K1) + noise
PSNR31 = metrics.peak_signal_noise_ratio(X2naive,b31)
MSE31 = metrics.mean_squared_error(X2naive, b31)
b32 = A(X2naive,K2) + noise
PSNR32 = metrics.peak_signal_noise_ratio(X2naive,b32)
MSE32 = metrics.mean_squared_error(X2naive, b32)

#Punto 3: Regolarizzazione

#Valore di lambda selezionato
_lambda = 0.5

def freg(x):
  J = x.reshape(m,n)
  res = 0.5*(np.linalg.norm(A(J, K)-b))**2 + (_lambda/2)*(np.linalg.norm(J))**2
  return np.sum(res)

def dfreg(x):
  J = x.reshape(m,n)
  res = AT(A(J, K)-b, K) + _lambda*J
  RES = np.reshape(res, m*n)
  return RES

def next_step(x,grad): # backtracking procedure for the choice of the steplength
  alpha=1.1
  rho = 0.5
  c1 = 0.25
  p=-grad
  j=0
  jmax=10
	#condizioni di Wolfe
  while (freg(x+alpha*p.reshape(m,n))> freg(x)+c1*alpha*grad.T@p  and j<jmax):
    alpha= rho*alpha #dimezzo alpha
    j+=1
  if (j>jmax): return -1
  else: return alpha #Se ritorna alpha allora la convergenza ad un punto stazionario è assicurata

def minimize2(x0): # funzione che implementa il metodo del gradiente
  k=1 #Numero di iterazioni
  MAXITERATIONS=1000 #Massimo numero di iterazioni
  ABSOLUTE_STOP=1.e-5 #soglia del gradiente
  x_last = x0 
  while (np.linalg.norm(dfreg(x_last))>ABSOLUTE_STOP and k < MAXITERATIONS ):
    k = k + 1
    grad = dfreg(x_last)#calcolo il gradiente
    x_last = x_last.reshape(m,n)
    step = next_step(x_last,grad) #Valore di alpha ritornato dalla funzione di backtracking
    x_last=x_last-step*grad.reshape(m,n) 
  return (x_last, k)

#risultato minimize scipy.optimize
result2 = minimize(freg,Xnaive,method='CG',jac=dfreg)
#risultato minimize implementata a lezione
result3 = minimize2(Xnaive)

Xreg = np.array([result2.x])
Xreg = Xreg.reshape(m,n)
X2reg = Xreg.astype(np.float64) / 255

b4 = A(X2reg,K) + noise
PSNR4 = metrics.peak_signal_noise_ratio(X2reg,b4)
MSE4 = metrics.mean_squared_error(X2reg, b4)

b41 = A(X2reg,K1) + noise
PSNR41 = metrics.peak_signal_noise_ratio(X2reg,b41)
MSE41 = metrics.mean_squared_error(X2reg, b41)

b42 = A(X2reg,K2) + noise
PSNR42 = metrics.peak_signal_noise_ratio(X2reg,b42)
MSE42 = metrics.mean_squared_error(X2reg, b42)

Xreg2 = np.array(result3[0])
Xreg2 = Xreg2.reshape(m,n)
X2reg2 = Xreg2.asype(np.float64) / 255

b5 = A(X2reg2,K) + noise
PSNR5 = metrics.peak_signal_noise_ratio(X2reg2,b5)
MSE5 = metrics.mean_squared_error(X2reg2, b5)

b51 = A(X2reg2,K1) + noise
PSNR51 = metrics.peak_signal_noise_ratio(X2reg2,b51)
MSE51 = metrics.mean_squared_error(X2reg2, b51)

b52 = A(X2reg2,K2) + noise
PSNR52 = metrics.peak_signal_noise_ratio(X2reg2,b52)
MSE52 = metrics.mean_squared_error(X2reg2, b52)

# Visualizziamo i risultati
plt.figure(figsize=(30, 30))
#Immagine originale
ax1 = plt.subplot(1, 4, 1)
ax1.imshow(X, cmap='gray', vmin=0, vmax=1)
plt.title('Immagine Originale')

#Nelle righe seguenti si gestiscono i vari livelli di corruzione dell'immagine, con i diversi blur e il noise applicato

#Corruzzione 1: Kernel 5 x 5 e sigma uguale a 0.5
ax2 = plt.subplot(1, 4, 2)
ax2.imshow(b, cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine Corrotta (PSNR: {PSNR:.2f}) (MSE: {MSE:.6f})')

#Corruzione 2: Kernel 7 x 7 e sigma uguale a 1
ax3 = plt.subplot(1, 4, 3)
ax3.imshow(b1, cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine Corrotta (PSNR: {PSNR1:.2f}) (MSE: {MSE1:.6f})')

#Corruzione 3: Kernel 9 x 9 e sigma uguale a 1.3
ax4 = plt.subplot(1, 4, 4)
ax4.imshow(b2, cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine Corrotta (PSNR: {PSNR2:.2f}) (MSE: {MSE2:.6f})')

plt.show()

plt.figure(figsize=(30, 30))

#Immagine minimizzata con soluzione naive K
ax5 = plt.subplot(1,3,1)
ax5.imshow(A(Xnaive,K),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con soluzione naive (PSNR: {PSNR3:.2f}) (MSE: {MSE3:.6f})')

#Immagine minimizzata con soluzione naive K1
ax6 = plt.subplot(1,3,2)
ax6.imshow(A(Xnaive,K1),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con soluzione naive (PSNR: {PSNR31:.2f}) (MSE: {MSE31:.6f})')

#Immagine minimizzata con soluzione naive K2
ax7 = plt.subplot(1,3,3)
ax7.imshow(A(Xnaive,K2),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con soluzione naive (PSNR: {PSNR32:.2f}) (MSE: {MSE32:.6f})')

plt.figure(figsize=(30, 30))

#Immagine minimizzata con regolarizzazione K
ax8 = plt.subplot(1,3,1)
ax8.imshow(A(Xreg,K),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con regolarizzazione (PSNR: {PSNR4:.2f}) (MSE: {MSE4:.6f})')

#Immagine minimizzata con regolarizzazione K1
ax9 = plt.subplot(1,3,2)
ax9.imshow(A(Xreg,K1),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con regolarizzazione (PSNR: {PSNR41:.2f}) (MSE: {MSE41:.6f})')

#Immagine minimizzata con regolarizzazione K2
ax10 = plt.subplot(1,3,3)
ax10.imshow(A(Xreg,K2),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con regolarizzazione (PSNR: {PSNR42:.2f}) (MSE: {MSE42:.6f})')

plt.show()

plt.figure(figsize=(30, 30))

#Immagine minimizzata con regolarizzazione 2
ax11 = plt.subplot(1,3,1)
ax11.imshow(A(Xreg2,K),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con regolarizzazione 2 (PSNR: {PSNR5:.2f}) (MSE: {MSE5:.6f})')

#Immagine minimizzata con regolarizzazione 2
ax12 = plt.subplot(1,3,2)
ax12.imshow(A(Xreg2,K1),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con regolarizzazione 2 (PSNR: {PSNR51:.2f}) (MSE: {MSE51:.6f})')

#Immagine minimizzata con regolarizzazione 2
ax13 = plt.subplot(1,3,3)
ax13.imshow(A(Xreg2,K2),cmap='gray', vmin=0, vmax=1)
plt.title(f'Immagine minimizzata con regolarizzazione 2 (PSNR: {PSNR52:.2f}) (MSE: {MSE52:.6f})')

plt.show()
