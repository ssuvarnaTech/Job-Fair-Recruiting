#paramters
import torch
import torch.nn as nn
import torch.nn.functional as F

#1d conv neural network



class CNN_NLP(nn.Module):
    def __init__(self, weights, vocab_size, sentence_length,keep_prob,
                 embed_dim = 300, num_classes = 2):
        super(CNN_NLP, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data = torch.tensor(weights)

        self.conv = nn.Conv2d(1, sentence_length, (4, 300))
        self.conv1 = nn.Conv2d(1, sentence_length, (4, 300))
        self.dropout = nn.Dropout(keep_prob)
        self.linLayer = nn.Linear(16, 2)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)
        activation = F.relu(conv_out.squeeze(3))
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)
        return max_out





    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings = embeddings.unsqueeze(1)
        # conv = self.conv(embeddings)
        # conv = F.relu(conv)
        # conv = conv.squeeze(3)
        max_out1 = self.conv_block(embeddings, self.conv)
        max_out2 = self.conv_block(embeddings, self.conv1)
        all_out = torch.cat((max_out1, max_out2), 1)
        fc_in = self.dropout(all_out)
        logits = self.linLayer(fc_in)
        # tags = self.linLayer(maxpooled)
        return logits
