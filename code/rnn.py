import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import pickle
import numpy as np
from sklearn.manifold import TSNE

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

        self.fc = nn.Linear(hidden_size * 2, len(util.relation_dict.keys()))

        if use_gpu == True:
            self.cuda()

    def forward(self, batch_sentence, batch_entity_position, batch_filler_position, max_len = None):
        sentence_len = list(map(len, self.util.batch_sentence_to_index(batch_sentence)))

        if max_len is None:
            batch_max_len = max(sentence_len)

        sentence_len = Variable(torch.FloatTensor(self.util.length_to_onehot(sentence_len, batch_max_len))).unsqueeze(2)

        if self.use_gpu == True:
            sentence_len = sentence_len.cuda()

        tmp_batch_size = sentence_len.size()[0] 

        sentence_embedding, reversed_sentence_embedding = self.embedding(batch_sentence, batch_entity_position, batch_filler_position)
        
        encoder_hidden_forward, encoder_cell_forward = self.init_encoder(tmp_batch_size)
        encoder_hidden_backward, encoder_cell_backward = self.init_encoder(tmp_batch_size)
        
        encoder_output_forward, (encoder_hidden_forward, encoder_cell_forward) = self.encoder_forward(sentence_embedding, (encoder_hidden_forward, encoder_cell_forward))
        encoder_output_backward, (encoder_hidden_backward, encoder_cell_backward) = self.encoder_backward(reversed_sentence_embedding, (encoder_hidden_backward, encoder_cell_backward))

        encoder_output = torch.cat([encoder_output_forward, encoder_output_backward], 2)

        encoder_output = torch.bmm(encoder_output.transpose(1,2), sentence_len).squeeze(2)

        output = nn.Softmax()(self.fc(encoder_output))

        return output


    def train(self, sentence, entity_position, filler_position, relation, batch_size, epoch, learning_rate, username,
            valid_sentence = None, valid_entity_position = None, valid_filler_position = None, valid_relation = None, max_len = None):
        optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate)

        loss_list = []

        for batch_sentence, batch_entity_position, batch_filler_position, batch_relation in self.util.batch_train(sentence, entity_position, filler_position, relation, batch_size, True):
            sentence_len = list(map(len, self.util.batch_sentence_to_index(batch_sentence)))

            if max_len is None:
                batch_max_len = max(sentence_len)

            sentence_len = Variable(torch.FloatTensor(self.util.length_to_onehot(sentence_len, batch_max_len))).unsqueeze(2)

            relation_index = self.util.batch_relation_to_index(batch_relation)
            relation_index = Variable(torch.LongTensor(relation_index))

            tmp_batch_size = sentence_len.size()[0] 

            if self.use_gpu == True:
                sentence_len = sentence_len.cuda()
                relation_index = relation_index.cuda()

            output = self(batch_sentence, batch_entity_position, batch_filler_position)

            loss = torch.nn.functional.cross_entropy(output, relation_index)
            loss /= tmp_batch_size

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_list.append(loss.cpu().data.numpy()[0])

        with open("../user/%s/model/loss" % username, "a") as f:
            f.write(str(np.mean(loss_list)))
            f.write("\n")

        with open("../user/%s/model/performance" % username, "a") as f:
            f.write("\t".join(map(str, self.F1_score(valid_sentence, valid_entity_position, valid_filler_position, valid_relation, batch_size, max_len))))
            f.write("\n")

        print(self.F1_score(valid_sentence, valid_entity_position, valid_filler_position, valid_relation, batch_size, max_len))

    def F1_score(self, sentence, entity_position, filler_position, relation, batch_size, max_len = None):
        output_list = []
        for batch_sentence, batch_entity_position, batch_filler_position in self.util.batch_forward(sentence, entity_position, filler_position, batch_size):
            output_list += self(batch_sentence, batch_entity_position, batch_filler_position, max_len).cpu().data


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
            P = 0
        R = correct / gold

        if (P + R) != 0:
            F1 = 2 * P * R / (P + R)
        else:
            F1 = 0

        return P, R, F1

    def init_encoder(self, batch_size):
        if self.use_gpu == True:
            return Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda(), Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(1, batch_size, self.hidden_size)), Variable(torch.zeros(1, batch_size, self.hidden_size))

    def save(self, username, epoch = 0):
        torch.save(self, "../user/%s/model/%d_%d_%d_latest" % (username, self.embedding.word_embedding.embedding_dim, self.embedding.entity_position_embedding.embedding_dim, self.hidden_size))
        if epoch != 0:
            torch.save(self, "../user/%s/model/%d" % (username, epoch))

    def visualize_data(self, sentence, entity_position, filler_position, relation, batch_size, username):
        data_list = []
        for batch_sentence, batch_entity_position, batch_filler_position, batch_relation in self.util.batch_train_discriminator(sentence, entity_position, filler_position, relation, batch_size, True):
            sentence_len = list(map(len, self.util.batch_sentence_to_index(batch_sentence)))

            batch_max_len = max(sentence_len)

            #relation_onehot = self.util.batch_relation_to_onehot(batch_relation)
            #no_relation_onehot = self.util.batch_no_relation_to_onehot(batch_relation)
            relation_index = self.util.batch_relation_to_index(batch_relation)

            sentence_len = Variable(torch.FloatTensor(self.util.length_to_onehot(sentence_len, batch_max_len))).unsqueeze(2)
            #relation_onehot = Variable(torch.FloatTensor(relation_onehot))
            #no_relation_onehot = Variable(torch.FloatTensor(no_relation_onehot))
            relation_index = Variable(torch.LongTensor(relation_index))

            tmp_batch_size = sentence_len.size()[0]

            #margin = Variable(torch.FloatTensor([[-0.7, -0.3]] * tmp_batch_size)) 

            if self.use_gpu == True:
                sentence_len = sentence_len.cuda()
                #relation_onehot = relation_onehot.cuda()
                #no_relation_onehot = no_relation_onehot.cuda()
                relation_index = relation_index.cuda()

                #margin = margin.cuda()

            sentence_embedding, reversed_sentence_embedding = self.embedding(batch_sentence, batch_entity_position, batch_filler_position)
           
            encoder_hidden_forward, encoder_cell_forward = self.init_encoder(tmp_batch_size)
            encoder_hidden_backward, encoder_cell_backward = self.init_encoder(tmp_batch_size)
           
            encoder_output_forward, (encoder_hidden_forward, encoder_cell_forward) = self.encoder_forward(sentence_embedding, (encoder_hidden_forward, encoder_cell_forward))
            encoder_output_backward, (encoder_hidden_backward, encoder_cell_backward) = self.encoder_backward(reversed_sentence_embedding, (encoder_hidden_backward, encoder_cell_backward))

            encoder_output = torch.cat([encoder_output_forward, encoder_output_backward], 2)

            encoder_output = torch.bmm(encoder_output.transpose(1,2), sentence_len).squeeze(2)
            
            data_list += encoder_output.data.numpy().tolist()

        data_list = TSNE(n_components=2).fit_transform(data_list)
        with open("../user/%s/figure/data_vis/latest.pkl" % username, "wb") as f:
            pickle.dump(data_list, f)
