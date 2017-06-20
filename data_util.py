import os
import numpy
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding


def encode_utf8(sentence):
    if isinstance(sentence, unicode):
        sentence = sentence.encode("ascii", "ignore").decode("ascii")
    if isinstance(sentence, str):
        sentence = sentence.decode("ascii", "ignore").encode("ascii")
    sentence = sentence.encode('utf-8')
    return sentence

def get_data(mode):
    path="/summary/neuralsum"
    all_document = []
    all_labels = []
    if mode == 'train':
        fname = ['dailymail/training/']
    if mode == 'validation':
        fname = ['dailymail/validation/']
    if mode == 'test':
        fname = ['dailymail/test/']

    pathname = []

    for f in fname:
        pathname.append(os.path.join(path, f))

    for path in pathname:
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            with open(filepath, 'r') as f:
                sentencelist = []
                labellist = []
                paragraphs = f.read().split('\n\n')
                text = encode_utf8(paragraphs[1])
                data = text.split('\n')

                for line in data:
                    line = line.strip()
                    line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ')

                    sentence, label = line.split('\t\t\t') 
                    if label == '2':
                        label = '0' 
                    sentencelist.append(sentence)
                    labellist.append(int(label))

                sentencelist = sentencelist[0:30]
                labellist = labellist[0:30]
                sentencelist += [" "] * (30 - len(sentencelist)) 
                labellist += [0] * (30 - len(labellist)) 
                
                all_document.append(sentencelist)
                all_labels.append(labellist)

    return all_document, all_labels


def build_vocab(all_document):
    vocab = []
    for document in all_document:
        for sentence in document:
            vocab.append(sentence)
    return vocab

def build_embeddingmatrix(X):
    vocab = build_vocab(X)

    print "Tokenizing input sequence"
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocab)
    word_index = tokenizer.word_index
    print "Found %s unique tokens." % len(word_index)


    for index, texts in enumerate(X):    
        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=50, dtype='int32',padding='post', truncating='post', value=0)
        X[index] = data

    embeddings_index = {}
    f = open(os.path.join('/glove/', 'glove.6B.100d.txt'))
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

    return embedding_vector, len(word_index)




