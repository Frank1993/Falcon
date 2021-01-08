import tensorflow as tf
from model import FalconModel
import numpy as np
def get_training_data():
    features = np.arange(100*3*5, dtype = np.int32).reshape(100,3,5)
    lables = np.zeros(100,dtype = np.float32).reshape([100,1])
    return features, lables

def train(num_steps):    
    with tf.Graph().as_default() as graph:
        features_placeholder = tf.placeholder(dtype = tf.int32, shape = [None, 3, 5])
        lables_placeholder = tf.placeholder(dtype = tf.float32, shape = [None, 1])
        
        falcon = FalconModel(10000,16, 10)
        
        logits = falcon.inference(features_placeholder)
        res = falcon.get_result(logits)
        loss = falcon.get_loss(logits,lables_placeholder)

        opt_op = tf.train.FtrlOptimizer(0.01).minimize(loss)
        init_op = tf. global_variables_initializer()
        
        
        with tf.Session() as sess:
            step = 0
            while step < num_steps:
                features,lables = get_training_data()
                sess.run(init_op)
                sess.run(opt_op, feed_dict = {features_placeholder:features, lables_placeholder:lables})
                step += 1


if __name__ == "__main__":
    train(100)
    