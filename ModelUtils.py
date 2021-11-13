from csv import reader
import csv
import math
import random

#read from word embeddings
def get_Vectors(file):
    word_to_index = dict()
    matrix = []
    f = open(file, "r")
    count = 0
    index = 0
    iteration = 1
    for line in f:
        vector = []
        if iteration == 1:
            iteration += 1
            continue
        if count == 100000:
            break
        lineSplit = line.split(" ")
        for x in range(1, len(lineSplit)):
            vector.append(float(lineSplit[x]))

        matrix.append(vector)
        word_to_index[lineSplit[0]] = index
        index += 1
        count += 1

    unk_vector = [random.randint(0, 1) for i in range(300)]
    pad_vector = [0 for i in range(300)]
    # if word not in matrix goes to unk which is for unknown words
    word_to_index["UNK"] = count
    word_to_index["PAD"] = count + 1
    matrix.append(unk_vector)
    matrix.append(pad_vector)
    return matrix, word_to_index


def read_data(file):
    labelsD = {"objective":0, "subjective":1}
    countS = 0
    countO = 0
    get_entries = []
    labels = []
    count = 0
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(count)
            print(row)
            if (len(row["sentence"]) == 0 and row["label"]) == '':
                continue
            if count == 2000:
                break
            get_entries.append(row["sentence"])
            labels.append(labelsD[row["label"]])
            if(labelsD[row["label"]] == 0):
                countO+=1
            else:
                countS +=1
            count += 2
        print('count O =', str(countO))
        print('count S =', str(countS))

    # retruns list with all the sentences without the labels
    return get_entries, labels


def get_average_sentence_length(list):
    sum = 0
    for sentence in list:
        sen = sentence.split(" ")
        sum += len(sen)

    average = sum / len(list)
    return math.floor(average)


def get_datasets(sentences, labels):
    # get train dataset
    length_of_train = int(0.8 * len(sentences))
    train_sentences = sentences[0: length_of_train]
    train_labels = labels[0: length_of_train]

    # get test dataset
    test_sentences = sentences[length_of_train:]
    test_labels = labels[length_of_train:]

    return train_labels, train_sentences, test_labels, test_sentences


def convert_sentences_to_vectors(train_sentences, test_sentences, average, word_to_index):
    matrix_train = []
    matrix_test = []
    for sentence in train_sentences:
        train_vector = []
        split_sentence = sentence.split(" ")
        if len(split_sentence) > average:
            split_sentence = split_sentence[0: average]

        for word in split_sentence:
            if word not in word_to_index:
                train_vector.append(word_to_index["UNK"])
            else:
                get_index_of_word = word_to_index[word]
                train_vector.append(get_index_of_word)
        if len(split_sentence) < average:
            for count in range(0, average - len(split_sentence)):
                train_vector.append(word_to_index["PAD"])
        matrix_train.append(train_vector)
    for sentence in test_sentences:
        test_vector = []
        split_sentence = sentence.split(" ")
        if len(split_sentence) > average:
            split_sentence = split_sentence[0: average]
        for word in split_sentence:
            if word not in word_to_index:
                test_vector.append(word_to_index["UNK"])
            else:
                get_index_of_word = word_to_index[word]
                test_vector.append(get_index_of_word)
        if len(split_sentence) < average:
            for count in range(0, average - len(split_sentence)):
                test_vector.append(word_to_index["PAD"])
        matrix_test.append(test_vector)
    return matrix_train, matrix_test


if __name__ == "__main__":
    sentences, labels = read_data('/Users/sreevanisuvarna/Documents/mySample(1).csv')
    # for s,l in zip(sentences, labels):
    #     print(s[0:5], l)
    print(get_average_sentence_length(sentences))
    train_labels, train_sentences, test_labels, test_sentences = get_datasets(sentences, labels)

