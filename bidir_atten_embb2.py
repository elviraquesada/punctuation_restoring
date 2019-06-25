
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, dot, concatenate, TimeDistributed, Bidirectional, Embedding
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 400  
epochs = 20  
start_index = 1
padding_index = 0



#SE PREPARAN LOS DATOS
def inputs():
    reader = pd.read_csv('G:/My Drive/Laura/frases_total2.csv',delimiter = ',', encoding = 'utf-8')
    sentences_without_punctuation =  pd.Series.tolist(reader.iloc[:,0])
    sentences_with_punctuation =  pd.Series.tolist(reader.iloc[:,1])
    sentences_without_punctuation=list(map(lambda x:x.lower(),sentences_without_punctuation))
    sentences_with_punctuation=list(map(lambda x:x.lower(),sentences_with_punctuation))
    return sentences_without_punctuation, sentences_with_punctuation

def encode_sequences(encoding_dict, sequences, max_length):
    encoded_sentence = np.zeros(shape=(len(sequences), max_length))
    for i in range(len(sequences)):
        for j in range(min(len(sequences[i]), max_length)):
            encoded_sentence[i][j] = encoding_dict[sequences[i][j]]
    return encoded_sentence

def build_dicts(input_data):
    encoding_dict = {}
    decoding_dict = {}
    for line in input_data:
        for char in line:
            if char not in encoding_dict:
                encoding_dict[char] = 2 + len(encoding_dict)
                decoding_dict[2 + len(decoding_dict)] = char
    
    return encoding_dict, decoding_dict, len(encoding_dict) + 2


input_texts, target_texts = inputs()


max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

data_size=len(input_texts)


input_encoding_dict, input_decoding_dict, input_dict_size = build_dicts(input_texts)
output_encoding_dict, output_decoding_dict, output_dict_size = build_dicts(target_texts)
print('output_dict_size:', output_dict_size)



encoded_input = encode_sequences(input_encoding_dict, input_texts, max_encoder_seq_length)
encoded_output = encode_sequences(output_encoding_dict, target_texts, max_decoder_seq_length)

decoder_input = np.zeros_like(encoded_output)
decoder_input[:, 1:] = encoded_output[:,:-1]
decoder_input[:, 0] = start_index
decoder_output = np.eye(output_dict_size)[encoded_output.astype('int')]


            
print('definimos el modelo')            
#SE DEFINE EL MODELO

encoder_inputs = Input(shape=(max_encoder_seq_length,))
encoder_embeddings = Embedding(input_dict_size, 128, input_length=max_encoder_seq_length, mask_zero=True)(encoder_inputs)
    
encoder = Bidirectional(LSTM(128, return_sequences=True, return_state=True, unroll=True), merge_mode='concat')(encoder_embeddings)
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder
encoder_h = concatenate([forward_h, backward_h])
encoder_c = concatenate([forward_c, backward_c])

decoder_inputs = Input(shape=(max_decoder_seq_length,))
decoder_embeddings = Embedding(output_dict_size, 256, input_length=max_decoder_seq_length, mask_zero=True)(decoder_inputs)

decoder_lstm = LSTM(256, return_sequences=True, unroll=True)
decoder_outputs = LSTM(256, return_sequences=True, unroll=True)(decoder_embeddings, initial_state=[encoder_h, encoder_c])

# luong attention
#alignment
attention = dot([decoder_outputs, encoder_outputs], axes=[2, 2])
attention = Activation('softmax', name='attention')(attention)
#context
context = dot([attention, encoder_outputs], axes=[2,1])

decoder_combined_context = concatenate([context, decoder_outputs])

output = TimeDistributed(Dense(128, activation="tanh"))(decoder_combined_context)
output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[output])

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# checkpoint
filepath="checkpoints/bidir_weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
early_stopping =EarlyStopping(monitor='val_loss', patience=2)
history=model.fit([encoded_input, decoder_input], decoder_output,
          batch_size=batch_size,
          epochs=epochs, callbacks=callbacks_list, 
          validation_split = 0.2)
print(history.history['loss'])
print(history.history['acc'])
print(history.history['val_loss'])
print(history.history['val_acc'])

#save model
model.save('model_bidirectional_attention_4.h5')

