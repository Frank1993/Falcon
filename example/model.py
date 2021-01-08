import tensorflow as tf
import feature_hash_ops
class FalconModel(object):
    """
    model definition of ctr prediction network
    """

    def __init__(self, numbuckets, size_per_key,feature_embedding_dim):
        """
        size_per_key:  key's real size in bytes
        """
        self.numbuckets = numbuckets
        self.size_per_key = size_per_key
        self.size_per_bucket = feature_hash_ops.get_size_per_bucket(self.size_per_key)
        print(self.size_per_bucket)
        self.feature_embedding_dim = feature_embedding_dim
    
    def inference(self, features):
        print("xxxxxxxxxxxx" + str(type(features)))
        # features dim -> [N, numfeatures, size_per_bucket/4]
        assert_op = tf.Assert(features.shape[-1] * 4 == self.size_per_bucket, [tf.shape(features)])
        with tf.control_dependencies([assert_op]):

            self.feature_hashtable = feature_hash_ops.initialize_feature_id_hashtable(self.numbuckets, self.size_per_key)
            
            # features_flat -> [N*numfeatures, size_per_bucket/4]
            features_flat = tf.reshape(features, [tf.shape(features)[0]*tf.shape(features)[1],-1])
            #features_flat = tf.Print(features_flat, [tf.shape(features_flat)],"feature flat shape")
            

            self.feature_ids = feature_hash_ops.hash_feature_to_id(self.feature_hashtable,features_flat, size_per_bucket = self.size_per_bucket, size_per_key=self.size_per_key)

            #self.feature_ids = tf.Print(self.feature_ids, [tf.shape(feature_ids)],"feature ids shape")

            self.feature_embedding_table = tf.get_variable(
                    "feature_embedding_table", [self.numbuckets, self.feature_embedding_dim], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.feature_embeddings = tf.gather(self.feature_embedding_table, self.feature_ids)
            
            #shape -> [N, D * feature_embedding_dim]
            self.features_embedding_input = tf.reshape(self.feature_embeddings, [tf.shape(features)[0],-1])

            
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=(3 * self.feature_embedding_dim,)))
            model.add(tf.keras.layers.Dense(32, activation='relu'))
            model.add(tf.keras.layers.Dense(16, activation='relu'))
            model.add(tf.keras.layers.Dense(1))
        

        return model(self.features_embedding_input)

    def get_loss(self, logits, labels):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)

    def get_result(self, logits):
        return tf.math.sigmoid(logits)




    