# fetch the data

files="
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
"

for f in $files
do
  dest=`basename $f`
  if [ ! -f $dest ] ; then
    wget $f
    gunzip -c $dest > ${dest%%.gz}
  fi
done

