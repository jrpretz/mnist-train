import numpy as np
import struct
import sys

def parse(category):
    label = None
    if category == "train":
        label = "train"
    if category == "test":
        label = "t10k"

    if label == None:
        raise RuntimeError( "Akkkkk")

    imagefile = open("%s-images-idx3-ubyte"%(label),"rb")

    magic_image, nitems_image = struct.unpack('>II',imagefile.read(8))
    rows, columns = struct.unpack('>II',imagefile.read(8))

    pixels = rows*columns

    # just a really strict test since its only one of two files I'm reading
    if(pixels != 784 or magic_image != 2051 ):
        raise RuntimeError( "Akkkkk")

    X = np.fromfile(imagefile, dtype=np.uint8, count=784*nitems_image)
    X = X.reshape(nitems_image,784)

    labelfile = open("%s-labels-idx1-ubyte"%(label),"rb")
    magic_label, nitems_label = struct.unpack('>II',labelfile.read(8))

    if( magic_label != 2049 or nitems_label != nitems_image):
        raise RuntimeError( "Akkkkk")

    labels=np.fromfile(labelfile, dtype=np.uint8, count=nitems_label)

    y = np.zeros(shape=(nitems_label,10))

    for i in range(0,nitems_label):
        y[i][labels[i]] = 1.0

    return (X,y)
