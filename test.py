import os
import cPickle
import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Activation
from keras.layers.pooling import AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.merge import Dot, Concatenate, Add
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from copy import deepcopy
from keras.utils import to_categorical
from nltk.tokenize import sent_tokenize


def get_word2id(mode):
    assert mode == 'test'
    word_index = cPickle.load(open('./jitender/word_index.p', 'rb'))
    return word_index

def build_embeddingmatrix(word_index):
    embeddings_index = {}
    f = open(os.path.join('../glove', 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = numpy.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def vectorize(X, word_idx):
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    for index, text in enumerate(X):
        sequences = tokenizer.texts_to_sequences(text)
        data = pad_sequences(sequences, maxlen=50, dtype='int32',padding='post', truncating='post', value=0)
        X[index] = data
        
    return numpy.array(X)


MAX_SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 200
MAX_SENTENCE = 30
EMBEDDING_DIM = 100
TIMESTAMP_1 = MAX_SEQUENCE_LENGTH
TIMESTAMP_2 = MAX_SENTENCE

word_index = get_word2id('test')
embedding_matrix = build_embeddingmatrix(word_index)

embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
main_input = Input(shape=(MAX_SENTENCE, MAX_SEQUENCE_LENGTH), dtype='float32', name="main_input")
sequence_input = TimeDistributed(embedding_layer, name="sequence_input")(main_input)
gru = GRU(HIDDEN_SIZE, return_sequences=True, kernel_initializer='glorot_uniform')
bi_gru = TimeDistributed(Bidirectional(gru, merge_mode='concat', weights=None), name = "bi_gru")(sequence_input)
pooled_hidden = TimeDistributed(AveragePooling1D(pool_size=TIMESTAMP_1, strides=None, padding='valid'), name="pooled_hidden")(bi_gru)
x = Reshape((TIMESTAMP_2, 2*HIDDEN_SIZE), name="x")(pooled_hidden)
h = Bidirectional(gru, merge_mode='concat', weights=None, name="h")(x)
dd = AveragePooling1D(pool_size=TIMESTAMP_2, strides=None, padding='valid', name="dd")(h)
d = Reshape((2*HIDDEN_SIZE,), name="d")(dd)

wc = K.variable(value=0.25, name="wc")
ws = K.variable(value=0.25, name="ws")
wr = K.variable(value=0.25, name="wr")

for j in range(MAX_SENTENCE):
    hj = Lambda((lambda X: X[:, j, :]), output_shape=(2*HIDDEN_SIZE,))(h)
    content = Dense(1, activation='sigmoid', name = "c"+str(j))(hj)
    salience = Dot(axes=1, normalize=True)([d, hj])
    salience = Activation('sigmoid')(salience)
    weighted_content = Lambda(lambda X: wc * X)(content)
    weighted_salience = Lambda(lambda X: ws * X)(salience)
    if j == 0:
        sj = Lambda((lambda X: 0 * X), output_shape=(2*HIDDEN_SIZE,))(hj)
        redundancy = Dot(axes=1, normalize=True)([hj, sj])
        redundancy = Activation('sigmoid')(redundancy)
        weighted_redundancy = Lambda(lambda X: ws * X)(redundancy)
        weighted_redundancy = Lambda(lambda X: -1 * X)(redundancy)
        score = Add()([weighted_content, weighted_salience, weighted_redundancy])
        p = Dense(2, activation='softmax', name = "p"+str(j))(score)
        prob = Reshape((1,2))(p)
    else:
        hj_minus_one = Lambda((lambda X: X[:, j-1, :]), output_shape=(2*HIDDEN_SIZE,))(h)
        pi_minus_one = Lambda((lambda X: X[:, j-1, 1]), output_shape=(1,))(prob)
        pi_minus_one = Reshape((1,))(pi_minus_one)
        sj_minus_one = Lambda(lambda X: pi_minus_one * X)(hj_minus_one)
        sj = Add()([sj_minus_one, sj])
        redundancy = Dot(axes=1, normalize=True)([hj, sj])
        redundancy = Activation('sigmoid')(redundancy)
        weighted_redundancy = Lambda(lambda X: ws * X)(redundancy)
        weighted_redundancy = Lambda((lambda X: -1 * X))(redundancy)
        score = Add()([weighted_content, weighted_salience, weighted_redundancy])
        p = Dense(2, activation='softmax', name = "p"+str(j))(score)
        p_prime = Reshape((1,2))(p)
        prob = Concatenate(axis=1)([prob, p_prime])

model = Model(inputs=main_input, outputs=prob)
model.load_weights("./jitender/model_weight_final.h5", by_name=True)


def encode_utf8(sentence):
    if isinstance(sentence, unicode):
        sentence = sentence.encode("ascii", "ignore").decode("ascii")
    if isinstance(sentence, str):
        sentence = sentence.decode("ascii", "ignore").encode("ascii")
    sentence = sentence.encode('utf-8')
    return sentence

def remove_punctuation(sentence):
    punctuation = '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'
    sentence = sentence.translate(string.maketrans("",""), punctuation)
    return sentence


text = open('Anil.txt').read()
text = encode_utf8(text)
text = text.replace('\n\n', ' ')
text = text.replace('\'', "")
sentencelist = sent_tokenize(text)

X = sentencelist

print  len(sentencelist), len(X)

XX = deepcopy(X)
X_vec = vectorize([XX], word_index)
y_pred = model.predict(X_vec, batch_size=1)

for idx, yy in enumerate(y_pred):    
    result = list(yy.argmax(1))

def calc_summary(X, y):
    assert len(X) == len(y)
    summary = ""
    for (sentence, label) in (zip(X, y)):
        if label == 1:
            summary = summary + " " + sentence

    return summary.strip()

summary = calc_summary(X, result)

print summary
print len(sent_tokenize(summary))