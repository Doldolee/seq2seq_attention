import tensorflow as tf
from utils import *

index_inputs, index_outputs, index_targets, data_configs = prepro_dataset()



class Encoder(tf.keras.layers.Layer):
  def __init__(self, **kargs):
    super(Encoder, self).__init__()
    self.enc_unit = kargs['units']
    self.batch = kargs['batch_sz']
    self.embedding = tf.keras.layers.Embedding(kargs['vocab_sz'], kargs['dim_embedding'])
    self.gru = tf.keras.layers.GRU(kargs['units'], return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

  ## input으로 들어가는 hidden은 layer의 처음 initail state를 넣어주는 것.(initialize_hidden_state함수)
  ## 근데 아마 없어도 되는것 같다.
  def initialize_hidden_state(self, inp):
    return tf.zeros((tf.shape(inp)[0], self.enc_unit))

  def call(self, x, hidden):
    x = self.embedding(x)
    hidden, output = self.gru(x) 
    return hidden, output

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self,**kargs):
    super(BahdanauAttention, self).__init__()
    self.Wc = tf.keras.layers.Dense(kargs['units'])
    self.Wb = tf.keras.layers.Dense(kargs['units'])
    self.Wa = tf.keras.layers.Dense(1)

  def call(self, output, hidden):
    #output의 차원을 hidden과 맞춰주기 위해 중간에 seq_len차원을 만들어준다.
    hidden_with_time_axis = tf.expand_dims(output, 1)

    output = self.Wc(output)
    hidden = self.Wb(hidden_with_time_axis)
    score = self.Wa(tf.nn.tanh(output + hidden))

    # axis를 잘 맞춰주어야한다.
    attention_weights = tf.nn.softmax(score, axis=1) 
    context_vector = attention_weights * hidden
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.layers.Layer):
  def __init__(self, **kargs):
    super(Decoder, self).__init__()

    self.embedding = tf.keras.layers.Embedding(kargs['vocab_sz'], kargs['units'])
    self.attention = BahdanauAttention(**kargs)
    self.gru = tf.keras.layers.GRU(kargs['units'], return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    self.fc = tf.keras.layers.Dense(kargs['vocab_sz'])


  def call(self, x, enc_hidden, enc_output):
    x = self.embedding(x)
    context_vector, attention_weights = self.attention(enc_output, enc_hidden)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    hidden, output = self.gru(x)

    hidden = tf.reshape(hidden, (-1, hidden.shape[2]))

    x = self.fc(hidden)

    return x, output, attention_weights


class seq2seq(tf.keras.Model):
  def __init__(self, **kargs):
    super(seq2seq, self).__init__()

    self.encoder = Encoder(**kargs)
    self.decoder = Decoder(**kargs)

    

  
  def call(self, x):
    inp, tar = x

    enc_hidden = self.encoder.initialize_hidden_state(inp)

    hidden, output = self.encoder(inp, enc_hidden)

    predict_tokens=list()
    # decodr입력을 sequence길이만큼 반복해서 각 단어를 넣어준다.
    for t in range(tar.shape[1]):
      # for문으로 한 번 먹었으므로 expand해주어야 (batch, 1)로 sequence 한 개가 입력된다.
      dec_input = tf.dtypes.cast(tf.expand_dims(tar[:,t],1), tf.float32)
      predictions, dec_hidden, _ = self.decoder(dec_input, hidden, output)

      predict_tokens.append(tf.dtypes.cast(predictions, tf.float32))

    return tf.stack(predict_tokens, axis=1)

  #batch 1인 경우만 구현
  def inference(self, x):
    inp = x
    hidden, output = self.encoder(inp)

    dec_input = tf.expand_dims([data_configs['std_symbol']],1)
    predict_tokens=list()
    for t in range(MAX_SEQUENCE):
      predictions, dec_hidden, _ = self.decoder(dec_input, hidden, output)

      predict_token = tf.argmax(predictions[0])

      ##아마 지금 inference하면 안돌아가고 아래 코드 주석풀고 위에 코드 지워야될 것임.
      # predict_token = tf.argmax(predictions, axis=2)
      # predict_token = tf.squeeze(predict_token)

      #end token idx
      if predict_token == 2: # 확인해봐야함.
        break
        
      predict_tokens.append(predict_token)
      dec_input = tf.dtypes.cast(tf.expand_dims([predict_token],0), tf.float32)

    return tf.stack(predict_tokens, axis=0).numpy()







    
    




    

