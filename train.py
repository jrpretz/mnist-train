import numpy as np
import tensorflow as tf
import sys
import h5py

import mnist_parse

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def forwardprop(X, w_1,b_1, w_2,b_2):
    h    = tf.nn.sigmoid(tf.matmul(X, w_1)+b_1)  # The \sigma function
    #yhat = tf.nn.sigmoid(tf.matmul(h, w_2)+b_2)  # The \varphi function
    yhat = tf.matmul(h, w_2)+b_2
    return yhat

def untrained_weights():
    w_1 = tf.Variable(tf.random_normal((784,50), stddev=0.1))
    b_1 = tf.Variable(tf.random_normal((1,50), stddev=0.1))
    w_2 = tf.Variable(tf.random_normal((50,10), stddev=0.1))
    b_2 = tf.Variable(tf.random_normal((1,10), stddev=0.1))
    return (w_1,b_1,w_2,b_2)

def trained_weights():
    weightfile = h5py.File("trained-weights.h5","r")
    
    w_1 = tf.Variable(np.array(weightfile["w_1"]))
    b_1 = tf.Variable(np.array(weightfile["b_1"]))
    w_2 = tf.Variable(np.array(weightfile["w_2"]))
    b_2 = tf.Variable(np.array(weightfile["b_2"]))
    return (w_1,b_1,w_2,b_2)

sess=tf.Session()


X = tf.placeholder(tf.float32 , shape=(None, 784),name="X")
y = tf.placeholder(tf.float32 , shape=(None, 10),name="y")

X_train,y_train = mnist_parse.parse("train")
X_test,y_test = mnist_parse.parse("test")



w_1,b_1,w_2,b_2 = untrained_weights()
#w_1,b_1,w_2,b_2 = trained_weights()

h = forwardprop(X,w_1,b_1,w_2,b_2)
pred = tf.nn.sigmoid(forwardprop(X,w_1,b_1,w_2,b_2))

cost    = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=h))
updates = tf.train.MomentumOptimizer(learning_rate=0.1,momentum=0.1).minimize(cost)
sess.run(tf.global_variables_initializer())
print(sess.run(cost,feed_dict={X: X_train, y: y_train}))

# if we want to do gradient descent on the WHOLE dataset
#for epoch in range(100):
#   sess.run(updates,feed_dict={X: X_train, y: y_train})
#   print("%d %f %f"%(epoch,sess.run(cost,feed_dict={X: X_train, y: y_train}),sess.run(cost,feed_dict={X: X_test, y: y_test})))

# if we wante to do gradient descent on just samples
for epoch in range(500):
    perm = np.random.permutation(y_train.shape[0])
    X_tmp = X_train[perm,:]
    y_tmp = y_train[perm,:]
    for i in range(int(y_train.shape[0]/1000)):
        low = i*1000
        high = (i+1)*1000
        sess.run(updates,feed_dict={X: X_tmp[low:high,:], y: y_tmp[low:high]})
    print("%d %f %f"%(epoch,sess.run(cost,feed_dict={X: X_train, y: y_train}),sess.run(cost,feed_dict={X: X_test, y: y_test})))


weights = {"w_1":sess.run(w_1),"w_2":sess.run(w_2),"b_1":sess.run(b_1),"b_2":sess.run(b_2)}

weightsFile = h5py.File('trained-weights.h5','w')
weightsFile.create_dataset("w_1",data=weights["w_1"])
weightsFile.create_dataset("b_1",data=weights["b_1"])
weightsFile.create_dataset("w_2",data=weights["w_2"])
weightsFile.create_dataset("b_2",data=weights["b_2"])

weightsFile.close()


p = sess.run(pred,feed_dict={X: X_test, y: y_test})
#for i in range(0,100):
    #print(p[i],y_test[i])
#    print(np.argmax(p[i]),np.argmax(y_test[i]))

p_labels = np.argmax(p,axis=1)
y_test_labels = np.argmax(y_test,axis=1)

#print(p_labels.shape,y_test_labels.shape)

import sklearn.metrics

print(sklearn.metrics.classification_report(p_labels,y_test_labels))
