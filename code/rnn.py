import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import pickle
import numpy as np

class Word_Embedding(nn.Module):
    def __init__(self, embedding_dim, util, use_gpu):
        super(Word_Embedding, self).__init__()

        self.embedding_dim = embedding_dim

        self.util = util
        self.vocab = util.vocab
        self.n_word = len(util.vocab.keys())

        self.use_gpu = use_gpu

        self.embedding = nn.Embedding(self.n_word, self.embedding_dim, padding_idx = util.vocab['<PAD>'])

        if use_gpu == True:
            self.cuda()

    def forward(self, sentence):
        sentence_index = self.util.batch_sentence_to_index(sentence)
        reversed_sentence_index = list(map(list, map(reversed, [i for i in sentence_index])))

        sentence_index = self.util.padding_sentence_index(sentence_index)
        reversed_sentence_index = self.util.padding_sentence_index(reversed_sentence_index)

        sentence_index = Variable(torch.LongTensor(sentence_index))
        reversed_sentence_index = Variable(torch.LongTensor(reversed_sentence_index))

        if self.use_gpu == True:
            sentence_index = sentence_index.cuda()
            reversed_sentence_index = reversed_sentence_index.cuda()

        return self.embedding(sentence_index), self.embedding(reversed_sentence_index)

class Position_Embedding(nn.Module):
    def __init__(self, embedding_dim, util, use_gpu):
        super(Position_Embedding, self).__init__()

        self.embedding_dim = embedding_dim
        
        self.util = util
        self.position_window_size = util.position_window_size
        
        self.use_gpu = use_gpu

        self.embedding = nn.Embedding((util.position_window_size * 2) + 2, self.embedding_dim, padding_idx = (util.position_window_size * 2) + 1)

        if use_gpu == True:
            self.cuda()

    def forward(self, position):
        position_index = self.util.batch_position_to_index(position)
        reversed_position_index = list(map(list, map(reversed, [i for i in position_index])))

        position_index = self.util.padding_position_index(position_index)
        reversed_position_index = self.util.padding_position_index(reversed_position_index)

        position_index = Variable(torch.LongTensor(position_index))
        reversed_position_index = Variable(torch.LongTensor(reversed_position_index))

        if self.use_gpu == True:
            position_index = position_index.cuda()
            reversed_position_index = reversed_position_index.cuda()

        return self.embedding(position_index), self.embedding(reversed_position_index)

class Sentence_Embedding(nn.Module):
    def __init__(self, word_embedding, entity_position_embedding, filler_position_embedding, util, use_gpu):
        super(Sentence_Embedding, self).__init__()

        self.word_embedding = word_embedding
        self.entity_position_embedding = entity_position_embedding
        self.filler_position_embedding = filler_position_embedding
        self.embedding_dim = self.word_embedding.embedding_dim + self.entity_position_embedding.embedding_dim + self.filler_position_embedding.embedding_dim

        self.util = util
        self.vocab = util.vocab

        self.use_gpu = use_gpu

        if self.use_gpu == True:
            self.cuda()

    def forward(self, sentence, entity_position, filler_position):
        word_embedding, reversed_word_embedding = self.word_embedding(sentence)
        entity_position_embedding, reversed_entity_position_embedding = self.entity_position_embedding(entity_position)
        filler_position_embedding, reversed_filler_position_embedding = self.filler_position_embedding(filler_position)

        sentence_embedding = torch.cat([word_embedding, entity_position_embedding, filler_position_embedding], 2)
        reversed_sentence_embedding = torch.cat([reversed_word_embedding, reversed_entity_position_embedding, reversed_filler_position_embedding], 2)

        return sentence_embedding, reversed_sentence_embedding

class Classifier(nn.Module):
    def __init__(self, sentence_embedding, hidden_size, util, use_gpu):
        super(Classifier, self).__init__()

        self.embedding_dim = sentence_embedding.embedding_dim
        self.hidden_size = hidden_size

        self.util = util
        self.vocab = util.vocab

        self.use_gpu = use_gpu

        self.embedding = sentence_embedding
        self.encoder_forward = nn.LSTM(self.embedding_dim, hidden_size, 1, batch_first = True)
        self.encoder_backward = nn.LSTM(self.embedding_dim, hidden_size, 1, batch_first = True)

        self.fc1 = nn.Linear(hidden_size * 2, 2)
        self.fc2 = nn.Linear(2, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, len(util.relation_dict.keys()))

        if use_gpu == True:
            self.cuda()

    def forward(self, batch_sentence, batch_entity_position, batch_filler_position, max_len = None):
        sentence_len_list = list(map(len, self.util.batch_sentence_to_index(batch_sentence)))

        if max_len is None:
            batch_max_len = max(sentence_len_list)

        sentence_len = Variable(torch.FloatTensor(self.util.length_to_onehot(sentence_len_list, batch_max_len))).unsqueeze(2)

        if self.use_gpu == True:
            sentence_len = sentence_len.cuda()

        tmp_batch_size = sentence_len.size()[0] 

        sentence_embedding, reversed_sentence_embedding = self.embedding(batch_sentence, batch_entity_position, batch_filler_position)
        
        encoder_hidden_forward, encoder_cell_forward = self.init_encoder(tmp_batch_size)
        encoder_hidden_backward, encoder_cell_backward = self.init_encoder(tmp_batch_size)
        
        encoder_output_forward, (encoder_hidden_forward, encoder_cell_forward) = self.encoder_forward(sentence_embedding, (encoder_hidden_forward, encoder_cell_forward))
        encoder_output_backward, (encoder_hidden_backward, encoder_cell_backward) = self.encoder_backward(reversed_sentence_embedding, (encoder_hidden_backward, encoder_cell_backward))

        encoder_output_list = []
        encoder_output = Variable(torch.FloatTensor(torch.zeros(tmp_batch_size, self.hidden_size * 2)))
        if self.use_gpu == True:
            encoder_output = encoder_output.cuda()
        
        for batch_idx in range(tmp_batch_size):
            encoder_index = Variable(torch.LongTensor(list(i for i in reversed(range(sentence_len_list[batch_idx])))))
            if self.use_gpu == True:
                encoder_index = encoder_index.cuda()

            encoder_output_list.append(torch.cat([encoder_output_forward[batch_idx][:sentence_len_list[batch_idx]], torch.index_select(encoder_output_backward[batch_idx], 0, encoder_index)], 1))
            encoder_output[batch_idx] = torch.cat([encoder_output_forward[batch_idx][sentence_len_list[batch_idx] - 1], encoder_output_backward[batch_idx][sentence_len_list[batch_idx] - 1]], 0)

        for batch_idx in range(tmp_batch_size):
            print(sentence_len_list[batch_idx])
            print(encoder_output_list[batch_idx].size())
        exit()

        att_list = []
        for batch_idx in range(tmp_batch_size):
            att_list.append()

        data_vis = self.fc1(encoder_output)
        output = nn.ReLU()(self.fc2(data_vis))
        output = nn.Softmax()(self.fc3(output))

        return output, data_vis


    def train(self, sentence, entity_position, filler_position, relation, label, batch_size, learning_rate, username,
            valid_sentence = None, valid_entity_position = None, valid_filler_position = None, valid_relation = None, max_len = None):

        optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate)

        loss_list = []

        for batch_sentence, batch_entity_position, batch_filler_position, batch_relation, batch_label in self.util.batch_train(sentence, entity_position, filler_position, relation, label, batch_size, True):
            sentence_len = list(map(len, self.util.batch_sentence_to_index(batch_sentence)))

            if max_len is None:
                batch_max_len = max(sentence_len)

            sentence_len = Variable(torch.FloatTensor(self.util.length_to_onehot(sentence_len, batch_max_len))).unsqueeze(2)

            relation_onehot = self.util.batch_relation_to_onehot(batch_relation)
            relation_onehot = Variable(torch.FloatTensor(relation_onehot))

            tmp_batch_size = sentence_len.size()[0] 

            batch_label = Variable(torch.FloatTensor(batch_label))

            if self.use_gpu == True:
                sentence_len = sentence_len.cuda()
                relation_onehot = relation_onehot.cuda()
                batch_label = batch_label.cuda()

            output = self(batch_sentence, batch_entity_position, batch_filler_position)[0]
            output = torch.bmm(output.unsqueeze(1), relation_onehot.unsqueeze(2)).squeeze()

            loss = torch.nn.BCELoss()(output, batch_label)
            loss /= tmp_batch_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.cpu().data.numpy()[0])

        with open("../user/%s/model/loss" % username, "a") as f:
            f.write(str(np.mean(loss_list)))
            f.write("\n")

        print(np.mean(loss_list))

    def test(self, sentence, entity_position, filler_position, relation, batch_size, username, max_len = None):
        with open("../user/%s/model/performance" % username, "a") as f:
            f.write("\t".join(map(str, self.F1_score(sentence, entity_position, filler_position, relation, batch_size, max_len))))
            f.write("\n")
        print(self.F1_score(sentence, entity_position, filler_position, relation, batch_size, max_len))

    def F1_score(self, sentence, entity_position, filler_position, relation, batch_size, max_len = None):
        output_list = []
        for batch_sentence, batch_entity_position, batch_filler_position in self.util.batch_forward(sentence, entity_position, filler_position, batch_size):
            output_list += self(batch_sentence, batch_entity_position, batch_filler_position, max_len)[0].cpu().data.numpy().tolist()
        output_list = np.array(output_list)

        thres = 0.5
        gold = 0
        out = 0
        correct = 0
        for i in range(len(output_list)):
            if relation[i] not in self.util.relation_dict.keys():
                out += sum(output_list[i] >= thres)
            else:
                if output_list[i][self.util.relation_to_index(relation[i])] >= thres:
                    correct += 1
                    out += 1
                gold += 1

        if out != 0:
            P = correct / out
        else:
            P = 0.0
        R = correct / gold

        if (P + R) != 0:
            F1 = 2 * P * R / (P + R)
        else:
            F1 = 0.0

        return P, R, F1

    def init_encoder(self, batch_size):
        if self.use_gpu == True:
            return Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda(), Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(1, batch_size, self.hidden_size)), Variable(torch.zeros(1, batch_size, self.hidden_size))

    def save(self, username, epoch = 0):
        torch.save(self, "../user/%s/model/%d_%d_%d_latest" % (username, self.embedding.word_embedding.embedding_dim, self.embedding.entity_position_embedding.embedding_dim, self.hidden_size))
        #if epoch != 0:
        #    torch.save(self, "../user/%s/model/%d" % (username, epoch))

    def visualize_data(self, sentence, entity_position, filler_position, relation, batch_size, username, max_len = None):
        data_dict = dict()
        data_list = []
        for batch_sentence, batch_entity_position, batch_filler_position in self.util.batch_forward(sentence, entity_position, filler_position, batch_size):
            data_list += self(batch_sentence, batch_entity_position, batch_filler_position, max_len)[1].cpu().data.numpy().tolist()

        for idx in range(len(relation)):
            if relation[idx] not in data_dict:
                data_dict[relation[idx]] = []
            data_dict[relation[idx]].append(data_list[idx])
    
        with open("../user/%s/figure/data_vis/data.pkl" % username, "wb") as f:
            pickle.dump(data_dict, f)
