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
from data_util import build_embeddingmatrix, get_data
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from copy import deepcopy
from data_util1 import get_dataa, build_matrix
from rouge import Rouge


rouge = Rouge()

def calc_rogue(X, y, y_pred):
    assert len(X) == len(y)
    assert len(X) == len(y_pred)
    rouge_1 = []
    rouge_2 = []
    rouge_l = []

    for i in range(len(X)):
        reference_summary = ""
        system_summary = ""
        for (sentence, label) in (zip(X[i], y[i])):
            if label == 1:
                reference_summary = reference_summary + " " + sentence

        for (sentence, label) in (zip(X[i], y_pred[i])):
            if label == 1:
                system_summary = system_summary + " " + sentence 
        
        reference_summary = reference_summary.strip()
        system_summary = system_summary.strip()
        score = rouge.get_scores(reference_summary, system_summary)
        rouge_1.append(score[0]['rouge-1']['f'])
        rouge_2.append(score[0]['rouge-2']['f'])
        rouge_l.append(score[0]['rouge-l']['f'])

    avg_r1 =  float(sum(rouge_1)) / len(rouge_1)
    avg_r2 =  float(sum(rouge_2)) / len(rouge_2)
    avg_rl =  float(sum(rouge_l)) / len(rouge_l)

    return avg_r1, avg_r2, avg_rl


def precision_recall(y_true, y_pred):
    p = []
    r = []
    f = []
    a = []
    for i in range(len(y_true)):
        p.append(precision_score(y_true[i], y_pred[i]))
        r.append(recall_score(y_true[i], y_pred[i]))
        f.append(f1_score(y_true[i], y_pred[i]))
        a.append(accuracy_score(y_true[i], y_pred[i]))
    
    avg_p =  float(sum(p)) / len(p)
    avg_r =  float(sum(r)) / len(r)
    avg_f =  float(sum(f)) / len(f)
    avg_a =  float(sum(a)) / len(a)
    
    return avg_p, avg_r, avg_f, avg_a


MAX_SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 200
MAX_SENTENCE = 30
EMBEDDING_DIM = 100
TIMESTAMP_1 = MAX_SEQUENCE_LENGTH
TIMESTAMP_2 = MAX_SENTENCE

X_test, y = get_data('test')
X = deepcopy(X_test)

embedding_matrix, len_word_index = build_embeddingmatrix(X_test)

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
model.load_weights('/model_weight/model_weight.h5', by_name=True)


X_val, y_val = get_dataa('validation')
X_val, y_val = X_val[0:5000], y_val[0:5000]
XX = deepcopy(X_val)

matrix, len_vector = build_matrix(X_val)
X_val = numpy.array(X_val)
y_pred = model.predict(X_val, batch_size=32)

result = []
for idx, yy in enumerate(y_pred):
    rslt = list(yy.argmax(1))
    result.append(rslt)

print "Prediction on test data completed"

r1, r2, rl = calc_rogue(XX, y_val, result)
print "ROUGE:"
print r1, r2, rl


p, r, f, acc = precision_recall(y_val, result)
print "PRECISION_RECALL:"
print p, r, f
print "ACCURACY:"
print acc