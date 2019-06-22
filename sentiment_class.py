import sklearn.metrics
import pandas as pd
import keras
import keras.layers
import keras.optimizers
import keras.utils
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, Dropout, SimpleRNN, Conv2D, Conv1D, Flatten, MaxPooling2D, MaxPooling1D, Embedding, LSTM, GRU, Bidirectional, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

read_csv_kwargs = dict(sep="\t",
                       converters={e: emotion_to_int.get for e in emotions})
train_data = pd.read_csv("2018-E-c-En-train.txt", **read_csv_kwargs)
dev_data = pd.read_csv("2018-E-c-En-dev.txt", **read_csv_kwargs)
test_data = pd.read_csv("2018-E-c-En-test.txt", **read_csv_kwargs)


# I consulted the following blog post for some guidance on the preprocessing step:
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/


t = Tokenizer(lower=True)
t.fit_on_texts(train_data['Tweet'])
word2id = t.word_index
id2word = {word2id[i]: i for i in word2id}
vocab_size = len(t.word_index) + 1
encoded_docs_train = t.texts_to_sequences(train_data['Tweet'])
encoded_docs_dev = t.texts_to_sequences(dev_data['Tweet'])
encoded_docs_test = t.texts_to_sequences(test_data['Tweet'])

# this is 36 on the train data, used to determine padding
max_doc_train = 0
for i in encoded_docs_train:
    if len(i) > max_doc_train:
        max_doc_train = len(i)

# max_doc_dev = 0
# for i in encoded_docs_dev:
#     if len(i) > max_doc_dev:
#         max_doc_dev = len(i)

x_train = pad_sequences(encoded_docs_train, maxlen=max_doc_train, padding='post')
y_train = np.zeros((len(encoded_docs_train), 11))
for row in range(len(train_data)):
    y_train[row] = train_data.iloc[row][emotions]


x_dev = pad_sequences(encoded_docs_dev, maxlen=max_doc_train, padding='post')
y_dev = np.zeros((len(encoded_docs_dev), 11))
for row in range(len(dev_data)):
    y_dev[row] = dev_data.iloc[row][emotions]

x_test = pad_sequences(encoded_docs_test, maxlen=max_doc_train, padding='post')


# takes about 5 mins and does NOT load everything, because there are words
# in word2id that do not appear in the GloVe vectors (16k vs. 11k words)
emb_index = {}
with open('glove.840B.300d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        if word in word2id:
            emb_index[word] = values[1:]

# this matrix will keep the zeros for the indices that do not appear in GloVe vectors
emb_matrix = np.random.uniform(low=-0.0001, high=0.0001, size=(vocab_size, 300))

for word in word2id:
    # there are about 10 words that have length 301 due to edge cases in line.split()
    if word in emb_index and len(emb_index[word]) != 301:
        emb_matrix[word2id[word]] = emb_index[word]




#CNN
# network = Sequential()
# network.add(Embedding(vocab_size, 300, weights=[emb_matrix], input_length=max_doc_train, trainable=False, mask_zero=True))
# network.add(Conv1D(filters=128, kernel_size=6))
# # network.add(MaxPooling1D())
# # network.add(Conv1D(filters=32, kernel_size=3))
# network.add(Flatten())
# network.add(BatchNormalization(axis=-1))
# network.add(Dense(units=50, activation='tanh'))
# network.add(Dropout(rate=0.1))
# network.add(Dense(units=25, activation='tanh'))
# network.add(Dropout(rate=0.1))
# network.add(Dense(units=11, activation='sigmoid'))
# network.compile(loss='binary_crossentropy', optimizer='sgd')
# network.fit(x=x_train, y=y_train, batch_size=16, epochs=30, verbose=1)

#RNN
network = Sequential()
network.add(Embedding(vocab_size, 300, weights=[emb_matrix], input_length=max_doc_train, trainable=False, mask_zero=True))
# network.add(SimpleRNN(units=128, activation='tanh'))
network.add(LSTM(units=128, activation='tanh'))
network.add(BatchNormalization(axis=-1))
network.add(Dense(units=50, activation='tanh'))
network.add(Dropout(rate=0.3))
network.add(Dense(units=25, activation='tanh'))
network.add(Dropout(rate=0.3))
network.add(Dense(units=11, activation='sigmoid'))
network.compile(loss='binary_crossentropy', optimizer='adam')
network.fit(x=x_train, y=y_train, batch_size=16, epochs=30, verbose=1)

vec = np.vectorize(int)

pred_train = vec(np.rint(network.predict(x_train)))
acc_train = sklearn.metrics.jaccard_similarity_score(pred_train, y_train)

pred_dev = vec(np.rint(network.predict(x_dev)))
acc_dev = sklearn.metrics.jaccard_similarity_score(pred_dev, y_dev)

pred_test = vec(np.rint(network.predict(x_test)))

print("train accuracy: %s" % acc_train)
print("dev accuracy: %s" % acc_dev)

dev_predictions = dev_data.copy()
dev_predictions[emotions] = pred_dev

test_predictions = test_data.copy()
test_predictions[emotions] = pred_test


if __name__ == "__main__":

    #saves predictions and prints out multi-label accuracy
    dev_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        dev_data[emotions], dev_predictions[emotions])))
