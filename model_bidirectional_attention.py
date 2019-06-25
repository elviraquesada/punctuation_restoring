'''
#Sequence to sequence example in Keras (character-level).

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

**Data download**

[English to French sentence pairs.
](http://www.manythings.org/anki/fra-eng.zip)

[Lots of neat sentence pairs datasets.
](http://www.manythings.org/anki/)

**References**

- [Sequence to Sequence Learning with Neural Networks
   ](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    ](https://arxiv.org/abs/1406.1078)
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, dot, concatenate, TimeDistributed, Bidirectional 
import numpy as np
import spacy
import re, string
import os
from keras.callbacks import ModelCheckpoint

def remove_punctuation (text):
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#import warnings; warnings.simplefilter('ignore')


# función auxiliar
def leer_texto(texto):
    """Funcion auxiliar para leer un archivo de texto"""
    curpath = os.path.abspath(os.curdir)
    with open(os.path.join(curpath, texto), 'r', encoding = 'utf-8') as text:
        return text.read()

def inputs():
    # Cargando el modelo en español de spacy
    nlp = spacy.load('es_core_news_md')
    
    # Procesando un texto 
    # Procesando 1984 de George Orwell - mi novela favorita

    texto_procesado1 = nlp(leer_texto('textos/con_puntuacion3.txt'))

    texto_procesado16 = nlp(leer_texto('textos/orwell.txt'))

    # Cuántas sentencias hay en el texto?
    sentencias1 = [s.string for s in texto_procesado1.sents]

    sentencias16 = [s.string for s in texto_procesado16.sents] 

    sentencias = sentencias1 + sentencias16
    sentencias_sin_puntuacion = [remove_punctuation(s) for s in sentencias]
    
    return sentencias_sin_puntuacion, sentencias

batch_size = 64  # Batch size for training.
epochs = 40  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.


# Vectorize the data.
input_characters = set()
target_characters = set()

input_texts, target_texts = inputs()

for input_text, target_text in zip(input_texts[: min(num_samples, len(input_texts) - 1)], target_texts[: min(num_samples, len(target_texts) - 1)] ):
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros(
    (len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros(
    (len(target_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,num_encoder_tokens))
encoder = Bidirectional(LSTM(128, return_sequences=True, return_state=True), merge_mode='concat')
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
state_h = concatenate([forward_h, backward_h])
state_c = concatenate([forward_c, backward_c])

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder = LSTM(latent_dim, return_sequences=True)
decoder = decoder(decoder_inputs, 
                                     initial_state=encoder_states)
# luong attention
attention = dot([decoder, encoder_outputs], axes=[2,2])
attention = Activation('softmax', name='attention')(attention)

context = dot([attention, encoder_outputs], axes=[2,1])

decoder_combined_context = concatenate([context, decoder])

output = TimeDistributed(Dense(128, activation="tanh"))(decoder_combined_context)
output = TimeDistributed(Dense(len(target_token_index), activation="softmax"))(output)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[output])
model.layers

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
# checkpoint
filepath="checkpoints/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs, callbacks=callbacks_list, 
          validation_split=0.2)
# Save model
model.save('model_bidirectional_attention.h5')

