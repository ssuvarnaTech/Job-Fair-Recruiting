import random

import torch
import torch.optim as optim
from torch import nn
import ModelUtils
from CNN import CNN_NLP
from sklearn import preprocessing
import torch
import torch.nn.functional as F


def train(train_x, train_y, model, epochs, batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for ep in range(epochs):
        temp = list(zip(train_x,train_y))
        random.shuffle(temp)
        train_x,train_y = zip(*temp)
        i = 0
        num_batches = 0
        running_loss = 0.0
        while (i < len(train_x)):
            if (i + batch_size < len(train_x)):
                inputs = train_x[i: i + batch_size]
                labels = train_y[i: i + batch_size]
            else:
                inputs = train_x[i:]
                labels = train_y[i:]
            inputs = torch.as_tensor(inputs)
            labels = torch.as_tensor(labels)
            optimizer.zero_grad()
            outputs = model(inputs)  # TODO: change cnn to model
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            i += batch_size
            num_batches += 1
            running_loss += loss.item()
            if num_batches % 50 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (ep + 1, num_batches + 1, running_loss / 50))
                running_loss = 0.0
        PATH = "model" + str(epochs) + ".ckpt"
        # TODO: save the model
        torch.save(model.state_dict(), PATH)


    print("Finished training")


def eval(model, PATH, test_sentences, test_labels):
    correct = 0
    total = 0
    model.load_state_dict(torch.load(PATH))
    model.eval()

    with torch.no_grad():
        for sentence, label in zip(test_sentences, test_labels):
            sentence_tensor = torch.as_tensor(sentence)
            sentence_tensor = sentence_tensor.unsqueeze(0)
            outputs = model(sentence_tensor)
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            if predicted == label:
                correct += 1

    print('Accuracy is: %d %%' % (100 * correct / total))


def main():
    # Call constructor and create model
    sentences, labels = ModelUtils.read_data('/Users/sreevanisuvarna/Documents/mySample (1).csv')
    train_labels, train_sentences, test_labels, test_sentences = ModelUtils.get_datasets(sentences, labels)
    # modified_train_labels = Label_Encode(train_labels)
    matrix, word_to_index = ModelUtils.get_Vectors('model.txt')

    train_sentences, test_sentences = ModelUtils.convert_sentences_to_vectors(train_sentences, test_sentences, 8,
                                                                              word_to_index)
    # print(len(train_sentences))
    # num = len(matrix[1])
    # print(num)
    cnn = CNN_NLP(matrix, 100000, 8, 0.5, embed_dim=300,
                  num_classes=2)  # weights, vocab_size, sentence_length, #TODO: pass params here
    train(train_sentences, train_labels, cnn, 100, 16)  # TODO: pass params here


    eval(cnn, "model10.ckpt", test_sentences, test_labels)

if __name__ == "__main__":
    main()
