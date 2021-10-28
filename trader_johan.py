import numpy as np
import pandas as pd
import os, datetime
import glob
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print('Tensorflow version: {}'.format(tf.__version__))
tf.config.list_physical_devices('GPU')

# cmd line to fix GPU lib path
# export LD_LIBRARY_PATH=~/Downloads/software/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=cuda/cuda/lib64:$LD_LIBRARY_PATH

import matplotlib.pyplot as plt
plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

################ Hyperparams ################
batch_size = 32
seq_len = 128

d_k = 256
d_v = 256
n_heads = 4
ff_dim = 256
epochs = 5

# steps (right now they're pretty small)
# epochs

input("Trader Johan, at your service. Code initialized. Press enter when ready to sail.") 
X_val, y_val = [], []
error_log = []
################ Load data ################

def load_data(filename):
  df = pd.read_csv(filename, delimiter=',', usecols=['open','high','low','close','volume','datetime'])

  # Replace 0 to avoid dividing by 0 later on
  df['volume'].replace(to_replace=0, method='ffill', inplace=True) 
  df.sort_values('datetime', inplace=True)
  df.tail()

  ###############################################################################
  '''Create indexes to split dataset'''

  times = sorted(df.index.values)
  last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
  last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series
  ###############################################################################
  '''Build data chicks then normalize'''

  train = split_train(df, last_20pct)
  val = split_val(df, last_20pct, last_10pct)
  test = split_test(df, last_10pct)

  train = normalize(train)
  val = normalize(val)
  test = normalize(test)

  train = pad_size(train)
  val = pad_size(val)
  test = pad_size(test)

  return train, val, test

def normalize(df):
  ################ Calculate normalized percentage change of all columns ################
  '''Calculate percentage change'''

  # df = (df[df['open'].str.contains(".")==True]) # fix here, remove try/catch around train loop

  df['open'] = df['open'].pct_change() # Create arithmetic returns column
  df['high'] = df['high'].pct_change() # Create arithmetic returns column
  df['low'] = df['low'].pct_change() # Create arithmetic returns column
  df['close'] = df['close'].pct_change() # Create arithmetic returns column
  df['volume'] = df['volume'].pct_change()

  df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

  ###############################################################################
  '''Normalize price columns'''
  
  min_return = min(df[['open', 'high', 'low', 'close']].min(axis=0))
  max_return = max(df[['open', 'high', 'low', 'close']].max(axis=0))

  # Min-max normalize price columns (0-1 range)
  df['open'] = (df['open'] - min_return) / (max_return - min_return)
  df['high'] = (df['high'] - min_return) / (max_return - min_return)
  df['low'] = (df['low'] - min_return) / (max_return - min_return)
  df['close'] = (df['close'] - min_return) / (max_return - min_return)

  ###############################################################################
  '''Normalize volume column'''

  min_volume = df['volume'].min(axis=0)
  max_volume = df['volume'].max(axis=0)

  # Min-max normalize volume columns (0-1 range)
  df['volume'] = (df['volume'] - min_volume) / (max_volume - min_volume)

  ###############################################################################

  return df

def pad_size(data):
    if len(data) < seq_len:   # <--- TODO this could be improved
      data = np.resize(data, (420, 5))
    else:
      data = data.values
    data = data[~np.isnan(data).any(axis=1)]
    return data

################ Create training, validation, and test data ################
def split_train(df, last_20pct):
  # Training data
  df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
  df_train.drop(columns=['datetime'], inplace=True)
  train_data = df_train
  #train_data = pad_size(train_data)
  return train_data

def build_train(train_data):
  X_train, y_train = [], []
  for i in range(seq_len, len(train_data)):
    X_train.append(train_data[i-seq_len:i]) # Chunks of training data with a length of 128 df-rows
    y_train.append(train_data[:, 3][i])
  X_train, y_train = np.array(X_train), np.array(y_train)
  print('Training set shape', X_train.shape, y_train.shape)
  return X_train, y_train, train_data

  ###############################################################################
def split_val(df, last_20pct, last_10pct):
  # Validation data
  df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
  df_val.drop(columns=['datetime'], inplace=True)
  val_data = df_val
  #val_data = pad_size(val_data)
  return val_data

def build_val(val_data):
  X_val, y_val = [], []
  for i in range(seq_len, len(val_data)):
      X_val.append(val_data[i-seq_len:i])
      y_val.append(val_data[:, 3][i])
  X_val, y_val = np.array(X_val), np.array(y_val)
  print('Validation set shape', X_val.shape, y_val.shape)
  return X_val, y_val, val_data

  ###############################################################################
def split_test(df, last_10pct):
  # Test data
  df_test = df[(df.index >= last_10pct)]
  df_test.drop(columns=['datetime'], inplace=True)
  test_data = df_test
  #test_data = pad_size(test_data)
  return test_data

def build_test(test_data):
  X_test, y_test = [], []
  for i in range(seq_len, len(test_data)):
      X_test.append(test_data[i-seq_len:i])
      y_test.append(test_data[:, 3][i])    
  X_test, y_test = np.array(X_test), np.array(y_test)
  print('Testing set shape' ,X_test.shape, y_test.shape)
  return X_test, y_test, test_data

################ Time Vector ################
class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

  def call(self, x):
    '''Calculate linear and periodic time features'''
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1) 
    time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config

################ Transformer ################
class SingleAttention(Layer):
  def __init__(self, d_k, d_v):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape): # why dense?
    self.query = Dense(self.d_k, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')
    
    self.key = Dense(self.d_k, 
                     input_shape=input_shape, 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='glorot_uniform')
    
    self.value = Dense(self.d_v, 
                       input_shape=input_shape, 
                       kernel_initializer='glorot_uniform', 
                       bias_initializer='glorot_uniform')

  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x/np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)
    
    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out    

#############################################################################

class MultiAttention(Layer):
  def __init__(self, d_k, d_v, n_heads):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))  
    
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7 
    self.linear = Dense(input_shape[0][-1], 
                        input_shape=input_shape, 
                        kernel_initializer='glorot_uniform', 
                        bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear   

#############################################################################

class TransformerEncoder(Layer):
  def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
    self.attn_dropout = Dropout(self.dropout_rate)  
    self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7 
    self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1) 
    self.ff_dropout = Dropout(self.dropout_rate)
    self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)    
  
  def call(self, inputs): # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer 

  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'d_k': self.d_k,
                   'd_v': self.d_v,
                   'n_heads': self.n_heads,
                   'ff_dim': self.ff_dim,
                   'attn_heads': self.attn_heads,
                   'dropout_rate': self.dropout_rate})
    return config

################ Model ################
def create_model():
  '''Initialize time and transformer layers'''
  time_embedding = Time2Vector(seq_len)
  attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
  attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

  '''Construct model'''
  in_seq = Input(shape=(seq_len, 5))
  x = time_embedding(in_seq)
  x = Concatenate(axis=-1)([in_seq, x])
  x = attn_layer1((x, x, x))
  x = attn_layer2((x, x, x))
  x = attn_layer3((x, x, x))
  x = GlobalAveragePooling1D(data_format='channels_first')(x)
  x = Dropout(0.1)(x)
  x = Dense(64, activation='relu')(x)
  x = Dropout(0.1)(x)
  out = Dense(1, activation='linear')(x)

  model = Model(inputs=in_seq, outputs=out)
  model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mape'])
  return model

def build_batches(data_path):
    while True:
        global X_val
        global y_val
        file_names = glob.glob(data_path + "/*.csv")
        for filename in file_names:
            #with np.load(file_name) as ticker:
          train, val, test = load_data(filename)
          X_train, y_train = build_train(train)[:2]
          X_val, y_val = build_val(val)[:2]
          yield (X_train, y_train)

def check_val_data(X_val, y_val, X_train, y_train):
  if X_val.size < 1:
    return X_train, y_train
  else:
    return X_val, y_val

#---------------------------------- Code Start ------------------------------------#

#data_path = './csvs'
#all_files = glob.glob(data_path + "/*.csv")
data_path = './csvs'
model = create_model()
model.summary()
all_files = glob.glob(data_path + "/*.csv")

callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
                                          monitor='val_loss', 
                                          save_best_only=False, verbose=1)
#tak = 1
for filename in all_files:
  if "^" not in filename:
    # Used for testing specific file
    '''if tak == 1:
      filename = './csvs/LINK.csv'
      tak = 0'''

    print(f"Loading {filename}. Progress: {all_files.index(filename) + 1}/{len(all_files)}")
    try:
      train, val, test = load_data(filename)
      X_train, y_train, train_data = build_train(train)
      X_val, y_val, val_data = build_val(val)
      X_test, y_test, test_data = build_test(test)

      history = model.fit(X_train, y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    callbacks=[callback],
                    validation_data=(check_val_data(X_val, y_val, X_train, y_train))) 
    except:
      error_log.append(filename)  # record which files were not trained with
      file = open("errors.txt", "w")
      str_error = repr(error_log)
      file.write(str_error + "\n")
      file.close()
      print(f"Error in {filename} has been logged.") 

    model = tf.keras.models.load_model('Transformer+TimeEmbedding.hdf5',
                                custom_objects={'Time2Vector': Time2Vector, 
                                                'SingleAttention': SingleAttention,
                                                'MultiAttention': MultiAttention,
                                                'TransformerEncoder': TransformerEncoder})

model = tf.keras.models.load_model('Transformer+TimeEmbedding.hdf5',
                                  custom_objects={'Time2Vector': Time2Vector, 
                                                  'SingleAttention': SingleAttention,
                                                  'MultiAttention': MultiAttention,
                                                  'TransformerEncoder': TransformerEncoder})

data_path = './csvs/A.csv' # Reassign arbitrary test set to see if prediction works
train, val, test = load_data(filename)
X_train, y_train, train_data = build_train(train)
X_val, y_val, val_data = build_val(val)
X_test, y_test, test_data = build_test(test)

###############################################################################
'''Calculate predictions and metrics'''

#Calculate predication for training, validation and test data
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

#Print evaluation metrics for all datasets
train_eval = model.evaluate(X_train, y_train, verbose=0)
val_eval = model.evaluate(X_val, y_val, verbose=0)
test_eval = model.evaluate(X_test, y_test, verbose=0)
print(' ')
print('Evaluation metrics')
print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

###############################################################################
'''Display results'''   # make sure I'm graphing the right stuff

fig = plt.figure(figsize=(15,20))
st = fig.suptitle("Transformer + TimeEmbedding Model", fontsize=22)
st.set_y(0.92)

#Plot training data results
ax11 = fig.add_subplot(311)
ax11.plot(train_data[:, 3], label='Closing Returns')
ax11.plot(np.arange(seq_len, train_pred.shape[0]+seq_len), train_pred, linewidth=3, label='Predicted Closing Returns')
ax11.set_title("Training Data", fontsize=18)
ax11.set_xlabel('Date')
ax11.set_ylabel('Closing Returns')
ax11.legend(loc="best", fontsize=12)

#Plot validation data results
ax21 = fig.add_subplot(312)
ax21.plot(val_data[:, 3], label='Closing Returns')
ax21.plot(np.arange(seq_len, val_pred.shape[0]+seq_len), val_pred, linewidth=3, label='Predicted Closing Returns')
ax21.set_title("Validation Data", fontsize=18)
ax21.set_xlabel('Date')
ax21.set_ylabel('Closing Returns')
ax21.legend(loc="best", fontsize=12)

#Plot test data results
ax31 = fig.add_subplot(313)
ax31.plot(test_data[:, 3], label='Closing Returns')
ax31.plot(np.arange(seq_len, test_pred.shape[0]+seq_len), test_pred, linewidth=3, label='Predicted Closing Returns')
ax31.set_title("Test Data", fontsize=18)
ax31.set_xlabel('Date')
ax31.set_ylabel('Closing Returns')
ax31.legend(loc="best", fontsize=12)
plt.savefig('results.pdf')

################ Model Metrics ################
'''Display model metrics'''

fig = plt.figure(figsize=(15,20))
st = fig.suptitle("Transformer + TimeEmbedding Model Metrics", fontsize=22)
st.set_y(0.92)

#Plot model loss
ax1 = fig.add_subplot(311)
ax1.plot(history.history['loss'], label='Training loss (MSE)')
ax1.plot(history.history['val_loss'], label='Validation loss (MSE)')
ax1.set_title("Model loss", fontsize=18)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend(loc="best", fontsize=12)

#Plot MAE
ax2 = fig.add_subplot(312)
ax2.plot(history.history['mae'], label='Training MAE')
ax2.plot(history.history['val_mae'], label='Validation MAE')
ax2.set_title("Model metric - Mean average error (MAE)", fontsize=18)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Mean average error (MAE)')
ax2.legend(loc="best", fontsize=12)

#Plot MAPE
ax3 = fig.add_subplot(313)
ax3.plot(history.history['mape'], label='Training MAPE')
ax3.plot(history.history['val_mape'], label='Validation MAPE')
ax3.set_title("Model metric - Mean average percentage error (MAPE)", fontsize=18)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Mean average percentage error (MAPE)')
ax3.legend(loc="best", fontsize=12)
plt.savefig('model_metrics.pdf')

'''
################ Model Architecture Overview ################
tf.keras.utils.plot_model(
    model,
    to_file="IBM_Transformer+TimeEmbedding.png",
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=96,)
'''