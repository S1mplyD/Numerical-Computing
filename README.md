# Numerical-Computing

[Numerical Computing](https://www.unibo.it/it/didattica/insegnamenti/insegnamento/2021/320581) project, blur and deblur of images. 

## Used methods

This program uses three methods to deblur an image: 
- Conjugate gradient method implemented using the [Scipy](https://scipy.org/) library
- Conjugate gradient method implemented using the [Scipy](https://scipy.org/) and Tikhonov regularization using a lambda value
- Gradient method implemented from scratch

## Requirements

Images MUST be 512 x 512 grayscale images!

### Libraries:

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [skimage](https://scikit-image.org/)
- [scipy](https://scipy.org/)

## Execution

Change the imagepath parameter with the path of your image and run the program. The program will use alle three methods described above and at the end of its executions will show you the currupted image and the reconstruction.
