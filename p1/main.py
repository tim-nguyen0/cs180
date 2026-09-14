#  CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import align as align

# name of the input file
imname = 'cathedral.jpg'

# read in the image
path = "data/"+imname
im = skio.imread(path)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)

    
# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# g_offset = align.calculate_offset_ncc(b, g)
# r_offset = align.calculate_offset_ncc(b, r)

g_offset = align.vectorized_calculate_offset_ncc(b, g)
r_offset = align.vectorized_calculate_offset_ncc(b, r)

rgb_aligned = align.align_and_crop(b, g, g_offset, r, r_offset)

# create a color image
im_out = sk.util.img_as_ubyte(np.dstack(rgb_aligned))

# save the image
fname = 'out/out_'+imname
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()
