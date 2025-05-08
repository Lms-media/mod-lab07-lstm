import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
 
from keras.src.models import Sequential
from keras.src.layers import Dense, Activation
from keras.src.layers import LSTM
 
from keras.src.optimizers import RMSprop
 
from keras.src.callbacks import LambdaCallback
from keras.src.callbacks import ModelCheckpoint
from keras.src.callbacks import ReduceLROnPlateau
import random

file = open('src/input.txt', encoding='utf-8')
text = file.read().lower().split()
    
vocabulary = sorted(list(set(text)))
 
word_to_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_to_word = dict((i, w) for i, w in enumerate(vocabulary))

max_length = 10
steps = 1
sentences = []
next_words = []

for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_words.append(text[i + max_length])
    
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_to_indices[word]] = 1
    y[i, word_to_indices[next_words[i]]] = 1
    
model = Sequential()
model.add(LSTM(128, input_shape =(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)

def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model.fit(X, y, batch_size = 128, epochs = 50)

def generate_text(length, diversity):
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated = sentence.copy()
    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_to_indices[word]] = 1.
 
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_word = indices_to_word[next_index]
 
            generated.append(next_word)
            sentence.append(next_word)
            sentence = sentence[1:]
    return ' '.join(generated)

generatedText = generate_text(1500, 0.2)

print(generatedText)
outputPath = os.path.join(os.path.dirname(__file__), '../result/gen.txt')
outputPath = os.path.normpath(outputPath)
with open(outputPath, 'w', encoding='utf-8') as f:
    f.write(generatedText)