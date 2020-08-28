import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_size = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences = True, return_state = True, recurrent_initializer = 'glorot_uniform')
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output ,state = self.gru(x, initial_state = hidden)
        return output, state
    
    def initialize_hidde_state(self):
        return tf.zeros((self.batch_size, self.enc_units))
    
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        #shape of query = (batch, hidden)
        #shape of values = (batch, max_len, hidden)
        query_with_time_axis = tf.expand_dims(query, axis=1)# shape = (batch, 1, hidden)
        
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))#shape = (batch, max_len, hidden)
        
        attention_weights = tf.nn.softmax(score, axis = 1)# shape = (batch, max_len, 1)
        
        context_vector = attention_weights * values # shape = (batch, max_len, hidden)
        context_vector = tf.reduce_sum(context_vector, axis = 1)# shape = (batch, 1, hidden)
        
        return context_vector, attention_weights
    

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dims, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size , embedding_dims)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences = True, return_state = True, recurrent_initializer = 'glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)
        
    def call(self, x, hidden, enc_output):
        #shape of enc_outputs = (batch, max_len, hidden)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) #shape = (batch_size, 1 , embedding_dims + hidden)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        
        return x, state, attention_weights