from __future__ import print_function
import numpy as np
import spacy
import re, string
import os
import pandas as pd
import textdistance
import datetime
from keras.models import load_model

char_start_encoding = 1
char_padding_encoding = 0


#SE PREPARAN LOS DATOS
def remove_punctuation_decoding (text):
    a = re.sub('[!?]', '.', text)
    b = re.sub('[…]', '...', a)
    e = re.sub('\n', '', b)
    d = re.sub('\xa0', '', e)
    c = re.sub('[!\"#$%&\'ºª“”’()­«»\*+\-–—:/\\<=>¡¿¨·?@[\]^_`´{\|}~]', '', d)
    return c

def remove_punctuation (text):
    a = re.sub('[!?]', '.', text)
    b = re.sub('[…]', '...', a)
    c = re.sub('\n', '', b)
    d = re.sub('\xa0', '', c)
    e = re.sub('[!\"#$%&\'ºª“”’()­«»\*+\-–—:/\\<=>¡¿¨·?@[\]^_`´{\|}~]', '', d)
    return re.sub('[%s]' % re.escape(string.punctuation), '', e)


# función auxiliar
def leer_texto(texto):
    """Funcion auxiliar para leer un archivo de texto"""
    curpath = os.path.abspath(os.curdir)
    with open(os.path.join(curpath, texto), 'r', encoding='utf-8') as text:
        return text.read()

def inputs_test():
    nlp = spacy.load('es_core_news_md')

    sentencias = []
    for i in range(80,81):
       texto_procesado = nlp(leer_texto("../textos/Output"+str(i)+".txt"))
       sentencias_texto =  [s.string for s in texto_procesado.sents]
       for sentencia in sentencias_texto:
           if len(sentencia) < 201 and len(sentencia)>10:
            sentencias.append(sentencia)

    sentencias_sin_puntuacion = [remove_punctuation(s) for s in sentencias]
    sentencias_con_puntuacion = [remove_punctuation_decoding(s) for s in sentencias]
    sentencias_sin_puntuacion=list(map(lambda x:x.lower(),sentencias_sin_puntuacion))
    sentencias_con_puntuacion=list(map(lambda x:x.lower(),sentencias_con_puntuacion))
    return sentencias_sin_puntuacion, sentencias_con_puntuacion

def inputs():
    reader = pd.read_csv('../frases_total2.csv',delimiter = ',', encoding = 'utf-8')
    sentencias_sin_puntuacion =  pd.Series.tolist(reader.iloc[:,0])
    sentencias_con_puntuacion =  pd.Series.tolist(reader.iloc[:,1])
    sentencias_sin_puntuacion=list(map(lambda x:x.lower(),sentencias_sin_puntuacion))
    sentencias_con_puntuacion=list(map(lambda x:x.lower(),sentencias_con_puntuacion))
    return sentencias_sin_puntuacion, sentencias_con_puntuacion

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


input_texts_test, target_texts_test = inputs_test()
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

         
print('definimos el modelo')            
#SE DEFINE EL MODELO

# checkpoint
model = load_model('model_bidirectional_attention4.h5')

print('modelo cargado')

#Greedy search
def generate(text):
    encoder_input = encode_sequences(input_encoding_dict, text, 200)
    decoder_input = np.zeros(shape=(len(encoder_input), 200))
    decoder_input[:,0] = char_start_encoding
    for i in range(1, max_decoder_seq_length):
        output = model.predict([encoder_input, decoder_input]).argmax(axis=2)
        decoder_input[:,i] = output[:,i]
    return decoder_input[:,1:]

def decode(decoding, sequence):
    text = ''
    for i in sequence:
        if i == 0:
            break
        text += output_decoding_dict[i]
    return text

def decode_text(text):
    print('vamos a decodificar')
    decoder_output = generate(text)
    return decode(output_decoding_dict, decoder_output[0])



count = 0
for seq_index in range(len(input_texts_test)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = input_texts_test[seq_index: seq_index + 1]
    expected_seq = target_texts_test[seq_index: seq_index + 1]
    input_seq = input_texts_test[0: 0 + 1]
    decoded_sentence = decode_text(input_texts_test[0: 0 + 1])
    print(input_seq)
    if textdistance.levenshtein.distance(decoded_sentence, target_texts_test[seq_index])==0:
        count+=1
    print('-')
    print('Input sentence:', input_texts_test[seq_index])
    print('Expected sentence:', target_texts_test[seq_index])
    print('Decoded sentence:', decoded_sentence)

a = datetime.datetime.now()
decoded_sentence = decode_text(['Colabore con nosotros Usted mismo ha dicho que vamos a modificar su propuesta'])

print('Input sentence:', 'Colabore con nosotros Usted mismo ha dicho que vamos a modificar su propuesta')
print('Expected sentence:', 'Colabore con nosotros. Usted mismo ha dicho que vamos a modificar su propuesta.')
b = datetime.datetime.now()
c = b-a
print('Decoded sentence:', decoded_sentence)