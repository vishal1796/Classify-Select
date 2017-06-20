import numpy
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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from copy import deepcopy
from data_util import build_embeddingmatrix, get_data
from custom_metrics import precision, recall, fmeasure


MAX_SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 200
MAX_SENTENCE = 30
EMBEDDING_DIM = 100
TIMESTAMP_1 = MAX_SEQUENCE_LENGTH
TIMESTAMP_2 = MAX_SENTENCE

X_train, y = get_data('train')
yi = deepcopy(y)
y_train = []
for i in range(len(y)):
    y_train.append(to_categorical(y[i], 2))


embedding_matrix, len_word_index = build_embeddingmatrix(X_train)
X_train, y_train, yi = numpy.array(X_train),  numpy.array(y_train),  numpy.array(yi)

embedding_layer = Embedding(len_word_index + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
main_input = Input(shape=(MAX_SENTENCE, MAX_SEQUENCE_LENGTH), dtype='float32', name="main_input")
sequence_input = TimeDistributed(embedding_layer, name="sequence_input")(main_input)
gru = GRU(HIDDEN_SIZE, return_sequences=True, kernel_initializer='glorot_uniform')
bi_gru = TimeDistributed(Bidirectional(gru, merge_mode='concat', weights=None), name = "bi_gru")(sequence_input)
pooled_hidden = TimeDistributed(AveragePooling1D(pool_size=TIMESTAMP_1, strides=None, padding='valid'), name="pooled_hidden")(bi_gru)
x = Reshape((TIMESTAMP_2, 2*HIDDEN_SIZE), name="x")(pooled_hidden)
h = Bidirectional(gru, merge_mode='concat', weights=None, name="h")(x)
dd = AveragePooling1D(pool_size=TIMESTAMP_2, strides=None, padding='valid', name="dd")(h)
d = Reshape((2*HIDDEN_SIZE,), name="d")(dd)

auxiliary_input = Input(shape=(MAX_SENTENCE,), name='aux_input')

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
        weighted_redundancy = Lambda((lambda X: -1 * X))(redundancy)
        score = Add()([weighted_content, weighted_salience, weighted_redundancy])
        p = Dense(2, activation='softmax', name = "p"+str(j))(score)
        prob = Reshape((1,2))(p)
    else:
        hj_minus_one = Lambda((lambda X: X[:, j-1, :]), output_shape=(2*HIDDEN_SIZE,))(h)
        yi_minus_one = Lambda((lambda X: X[:, j-1]), output_shape=(1,))(auxiliary_input)
        yi_minus_one = Reshape((1,))(yi_minus_one)
        sj_minus_one = Lambda(lambda X: yi_minus_one * X)(hj_minus_one)
        sj = Add()([sj_minus_one, sj])
        redundancy = Dot(axes=1, normalize=True)([hj, sj])
        redundancy = Activation('sigmoid')(redundancy)
        weighted_redundancy = Lambda(lambda X: ws * X)(redundancy)
        weighted_redundancy = Lambda((lambda X: -1 * X))(redundancy)
        score = Add()([weighted_content, weighted_salience, weighted_redundancy])
        p = Dense(2, activation='softmax', name = "p"+str(j))(score)
        p_prime = Reshape((1,2))(p)
        prob = Concatenate(axis=1)([prob, p_prime])


model = Model(inputs=[main_input, auxiliary_input], outputs=prob)
model.compile(loss='binary_crossentropy', metrics=['accuracy', precision, recall, fmeasure], optimizer=Adam())

# filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpointer = ModelCheckpoint(filepath, monitor = 'val_acc', verbose=0, save_best_only = True)
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
# callbacks_list = [checkpointer, early_stop]

model.fit([X_train, yi], y_train, epochs=50, batch_size=32, verbose=1, validation_split = 0.1)
model.save_weights("/output/model_weight_final.h5")
