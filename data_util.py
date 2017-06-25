import os
import numpy

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