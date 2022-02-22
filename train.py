from utils import *
import tensorflow as tf
from model import *
from loss import *

index_inputs, index_outputs, index_targets, data_configs = prepro_dataset()

kargs={
  "batch_sz":2,
  "vocab_sz" : data_configs['vocab_size'],
  "dim_embedding" : 4,
  "units" : 8,
  "EPOCH" : 30,
  "MAX_SEQUENCE" : 25

}


model = seq2seq(**kargs)
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.001), metrics=[accuracy])

checkpoint_path = "./weights.h5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True)

earlystop_cb=tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10)

history = model.fit([index_inputs, index_outputs], index_targets, batch_size=kargs['batch_sz'], epochs=kargs['EPOCH'], validation_split=0.1, callbacks=[earlystop_cb, cp_callback])


