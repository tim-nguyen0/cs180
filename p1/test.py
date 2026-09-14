import numpy as np
import skimage as sk
import skimage.io as skio
import align as align
import pyramid as pyramid

# name of the input file
imname = 'church.tif'

# read in the image
path = "data/"+imname
im = skio.imread(path)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
im = pyramid.image_pyramid(im, 5)

skio.imshow(im[0])
skio.show()