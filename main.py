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
image = io.imread('./imagecalc7.png')  

#Definisco il tipo dell'immagine e normalizzo l'immagine
Xoriginal = image.astype(np.float64) / 255.0
m, n = Xoriginal.shape

# Genera i diversi filtri di blur
#K= psf_fft(gaussian_kernel(5, 0.5), 5, Xoriginal.shape)
#K = psf_fft(gaussian_kernel(7, 1), 7, Xoriginal.shape)
K = psf_fft(gaussian_kernel(9, 1.3), 9, Xoriginal.shape)

# Genera il rumore
dev = 0.01218556955205374#np.random.uniform(low=0.0, high=0.05)
noise = np.random.normal(size=Xoriginal.shape) * dev

# Aggiungi i vari livelli di blur e il rumore, per generare diverse immagini corrotte
b = A(Xoriginal, K) + noise

ATb = AT(b,K)

#Calcolo del PSNR tra immagine originale e corrotta
PSNR = metrics.peak_signal_noise_ratio(Xoriginal, b)

#Calcolo MSE tra immagine reale e corrotta
MSE = metrics.mean_squared_error(Xoriginal, b)

#Punto 2: Soluzione naive
k = 0

file = open("results.txt", "w")
fVal = np.zeros((1,1000))
dfVal = np.zeros((1,1000))
err_norm_f = np.zeros((1,1000))

def countIt(x):
  global k
  k +=1

#Funzione che è necessario minimizzare per ottenere il risultato con soluzione naive
def f(x):
  J = x.reshape(m,n)
  res = 0.5*(np.linalg.norm(A(J, K)-b,ord=2))**2
  fVal[:,k] = res
  return res

#Funzione per il calcolo del gradiente
def df(x):
  J = x.reshape(m,n)
  res = AT(A(J, K), K) - ATb
  # print(f"norma gradiente df: {res}")
  err_norm_f[:,k]=(np.linalg.norm(J-b))
  RES = np.reshape(res, m*n)
  dfVal[:,k]=(np.linalg.norm(RES))
  return RES


x0 = np.zeros(m*n)
#Immagine ricostruita con metodo naive Kernel 5x5
naive = minimize(f,x0,method='CG',callback=countIt,jac=df)
dfVal = dfVal[:,:k]
err_norm_f = err_norm_f[:,:k]
fVal = fVal[:,:k]
k = 0
file.write("iterazioni naive: ")
file.write(str(naive.nit))
file.write("\n")
plt.plot(dfVal.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Norma Gradiente')
plt.title('Iterazioni vs Norma Gradiente')
plt.grid()
plt.show()
plt.plot(err_norm_f.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show()
plt.plot(fVal.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Funzione Obiettivo')
plt.title('Iterazioni vs Funzione Obiettivo')
plt.grid()
plt.show()

#immagini ricostruite
Xnaive = np.array([naive.x])

Xnaive = Xnaive.reshape(m,n)

#Calcolo PSNR e MSE tra immagine originale e ricostruita
PSNRnaive = metrics.peak_signal_noise_ratio(Xoriginal,Xnaive)
MSEnaive = metrics.mean_squared_error(Xoriginal, Xnaive)

#Punto 3: Regolarizzazione

#Valore di lambda selezionato
_lambda = 0.04
fregVal = np.zeros((1,1000))
dfregVal = np.zeros((1,1000))
err_norm_freg = np.zeros((1,1000))

def freg(x):
  J = x.reshape(m,n)
  res = 0.5*(np.linalg.norm(A(J, K)-b))**2 + (_lambda/2)*(np.linalg.norm(J))**2
  fregVal[:,k] = res
  return res

def dfreg(x):
  J = x.reshape(m,n)
  res = AT(A(J, K), K) - ATb + _lambda*J
  # print(f"norma gradiente dfreg: {res}")
  err_norm_freg[:,k]=(np.linalg.norm(J-b))
  RES = np.reshape(res, m*n)
  dfregVal[:,k]=(np.linalg.norm(RES))
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
  k=-1 #Numero di iterazioni
  MAXITERATIONS=1000 #Massimo numero di iterazioni
  ABSOLUTE_STOP=1.e-5 #soglia del gradiente
  norm_grad_list=np.zeros((1,MAXITERATIONS))
  function_eval_list=np.zeros((1,MAXITERATIONS))
  error_list=np.zeros((1,MAXITERATIONS))
  x_last = x0 
  while (np.linalg.norm(dfreg(x_last))>ABSOLUTE_STOP and k < MAXITERATIONS ):
    k = k + 1
    grad = dfreg(x_last)#calcolo il gradiente
    x_last = x_last.reshape(m,n)
    step = next_step(x_last,grad) #Valore di alpha ritornato dalla funzione di backtracking
    x_last=x_last-step*grad.reshape(m,n) 
    function_eval_list[:,k]=freg(x_last)
    error_list[:,k]=np.linalg.norm(x_last-b)
    norm_grad_list[:,k]=np.linalg.norm(dfreg(x_last))
  function_eval_list = function_eval_list[:,:k]
  error_list = error_list[:,:k]
  norm_grad_list = norm_grad_list[:,:k]
  return (x_last, k, function_eval_list,error_list,norm_grad_list)

#risultato minimize gradiente coniugato
regCG = minimize(freg,x0,callback=countIt,method='CG',jac=dfreg)
dfregVal = dfregVal[:,:k]
err_norm_freg = err_norm_freg[:,:k]
fregVal = fregVal[:,:k]
k = 0
plt.plot(dfregVal.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Norma Gradiente')
plt.title('Iterazioni vs Norma Gradiente')
plt.grid()
plt.show()
plt.plot(err_norm_freg.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show()
plt.plot(fregVal.T, 'o-')
plt.xlabel('iter')
plt.ylabel('Funzione Obiettivo')
plt.title('Iterazioni vs Funzione Obiettivo')
plt.grid()
plt.show()

#funzione
# plt.plot(np.arange((len(fregVal))),fregVal,color="red",linestyle="--")
# plt.show()
# #norma
# plt.plot(np.arange((len(dfregVal))),dfregVal,color="red",linestyle="--")
# plt.show()
#_lambda = 0.6
# regCG1 = minimize(freg,x0,method='CG',jac=dfreg)
# iter = np.arange((len(fregVal)))
# plt.plot(iter,fregVal,color="red",linestyle="--")
# plt.ylabel("valore funzione")
# plt.show()
# fregVal.clear()
#risultato minimize gradiente implementato a lezione
regG = minimize2(x0)
file.write("iterazioni regG: ")
file.write(str(regG[1]))
file.write("\n")
file.close()
print(regG[4].T)
plt.plot(regG[4].T, 'o-')
plt.xlabel('iter')
plt.ylabel('Norma Gradiente')
plt.title('Iterazioni vs Norma Gradiente')
plt.grid()
plt.show()
plt.plot(regG[3].T, 'o-')
plt.xlabel('iter')
plt.ylabel('Errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show()
plt.plot(regG[2].T, 'o-')
plt.xlabel('iter')
plt.ylabel('Funzione Obiettivo')
plt.title('Iterazioni vs Funzione Obiettivo')
plt.grid()
plt.show()
# plt.plot(np.arange((len(fregVal))),fregVal,color="green",linestyle="--")
# plt.show()
# #norma
# plt.plot(np.arange((len(dfregVal))),dfregVal,color="green",linestyle="--")
# plt.show()

#immagini ricostruite

XregCG = np.array([regCG.x])

XregCG = XregCG.reshape(m,n)

XregG = np.array(regG[0])

XregG = XregG.reshape(m,n)

#Calcolo PSNR e MSE tra immagine originale e immagine ricreata tramite metodo gradienti coniugati
PSNRregCG = metrics.peak_signal_noise_ratio(Xoriginal,XregCG)
MSEregCG = metrics.mean_squared_error(Xoriginal, XregCG)

#Calcolo PSNR e MSE tra immagine originale e immagine ricreata tramite metodo del gradiente
PSNRregG = metrics.peak_signal_noise_ratio(Xoriginal,XregG)
MSEregG = metrics.mean_squared_error(Xoriginal, XregG)

# # Visualizziamo i risultati
# plt.figure(figsize=(30, 30))
# #Immagine originale
# ax1 = plt.subplot(1, 2, 1)
# ax1.imshow(Xoriginal, cmap='gray', vmin=0, vmax=1)
# plt.title('Immagine Originale')

# #Immagine corrotta
# ax2 = plt.subplot(1, 2, 2)
# ax2.imshow(b, cmap='gray', vmin=0, vmax=1)
# plt.title(f'Immagine Corrotta (PSNR: {PSNR:.2f}) (MSE: {MSE:.6f})')

# plt.show()

# plt.figure(figsize=(30, 30))
# #Immagine minimizzata con soluzione naive 
# ax5 = plt.subplot(1,3,1)
# ax5.imshow(Xnaive,cmap='gray', vmin=0, vmax=1)
# plt.title(f'Immagine minimizzata con soluzione naive (PSNR: {PSNRnaive:.2f}) (MSE: {MSEnaive:.6f})')

# #Immagine minimizzata con regolarizzazione 
# ax8 = plt.subplot(1,3,2)
# ax8.imshow(XregCG,cmap='gray', vmin=0, vmax=1)
# plt.title(f'Immagine minimizzata con regolarizzazione (PSNR: {PSNRregCG:.2f}) (MSE: {MSEregCG:.6f})')

# #Immagine minimizzata con regolarizzazione 2
# ax11 = plt.subplot(1,3,3)
# ax11.imshow(XregG,cmap='gray', vmin=0, vmax=1)
# plt.title(f'Immagine minimizzata con regolarizzazione 2 (PSNR: {PSNRregG:.2f}) (MSE: {MSEregG:.6f})')

# plt.show()

# print(f"PSNR: {PSNR,PSNRnaive,PSNRregCG,PSNRregG}")
# print(f"MSE: {MSE,MSEnaive,MSEregCG,MSEregG}")
# print(f"dev: {dev}")
