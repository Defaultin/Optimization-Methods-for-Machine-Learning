# # Exercise 7.1 Neural Network for wine classification
# Goals: Build a one hidden layer dense neural network and see how sessions work.
!pip install tensorflow==1.6.0


import tensorflow as tf
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split as tts

# Load data
wine = datasets.load_wine()
wine_feats = wine.data # 178 datapoints x 13 feats
wine_labels = wine.target # 3 categories

# Preprocess feats (normalization)
scaler = preprocessing.StandardScaler()
wine_feats = scaler.fit_transform(wine_feats)

# Preprocess labels (transform integer label to corresponding unit vectors)
one_hot_enc = preprocessing.OneHotEncoder(categories='auto')
wine_labels = one_hot_enc.fit_transform(wine_labels.reshape(-1, 1)).todense() # 178 datapoints x 3 categories

# split data
train_feats, test_feats, train_labels, test_labels = tts(wine_feats, wine_labels, test_size=8)

num_input = 13 # number of feats
num_output = 3 # in 3dim unit vectors converted labels

# Meta Parameter, insert a different number if you want to
num_layer1 = 200 # number of neurons in your hidden layer 


tf.reset_default_graph()

# Build your computation graph below
# Start with defining placeholders and optimization variables (weights and biases)

x = tf.placeholder(tf.float32, [num_input, None], name='x')
y = tf.placeholder(tf.float32, [num_output, None], name='y')

A1 = tf.Variable(tf.random_normal([num_layer1, num_input], seed=1), name='A1') 
b1 = tf.Variable(tf.random_normal([num_layer1, 1], seed=1), name='b1')
l1 = tf.nn.sigmoid(tf.add(tf.matmul(A1, x), b1)) # Hidden Layer

A2 = tf.Variable(tf.random_normal([num_output, num_layer1], seed=1), name='A2') 
b2 = tf.Variable(tf.random_normal([num_output, 1], seed=1), name='b1')
out = tf.add(tf.matmul(A2, l1), b2) # Output Layer

pred = tf.nn.softmax(out, axis=2) # Predictions

# Use for example one of those losses
loss = tf.losses.absolute_difference(pred, y,reduction=tf.losses.Reduction.MEAN) 
#loss = tf.losses.softmax_cross_entropy(logits=out, onehot_labels=y)

# Can use different optimizers
#optimizer = tf.train.AdamOptimizer().minimize(loss) 
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options={'maxiter': 10})


# You may wanna have a look at your graph
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())


# Run a session to train your network and calculate ratio of succes
iterations = 500
with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Choose one of the two variants below!
        # Use this loop for Tensorflow optimizers like GradientDescent or Adam
        #for j in range(iterations): 
        #    p, l, _ = sess.run([pred, loss, optimizer], feed_dict={x: train_feats.T, y: train_labels.T})
        
        # Use this for Scipy BFGS optimizer
        optimizer.minimize(sess, feed_dict={x: train_feats.T, y: train_labels.T})
        p, l = sess.run([pred, loss], feed_dict={x: train_feats.T, y: train_labels.T})
        
        print('ROS on train set:', np.mean(np.argmax(train_labels, axis=1).T == np.argmax(p.T,axis=1)))
        
        test_lab_opt = sess.run(pred, feed_dict={x: test_feats.T}).T
        
        print('ROS on test set:', np.mean(np.argmax(test_labels, axis=1).T == np.argmax(test_lab_opt,axis=1)))