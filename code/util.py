import numpy as np
from collections import Counter
import multiprocessing as mp
import copy

class Util():
    def __init__(self, sentence, relation_list_path, vocabulary_threshold, position_window_size):
        self.vocabulary_threshold = vocabulary_threshold

        self.vocab, self.reverse_vocab = self.build_vocab(sentence, vocabulary_threshold)

        self.relation_dict = self.build_relation_dict(relation_list_path)
        
        self.position_window_size = position_window_size

    def build_vocab(self, sentence, vocab_thres):
        word_list = " ".join(sentence).split(" ")
        word_count = Counter(word_list)
            
        for word in list(word_count.keys()):
            if word_count[word] < vocab_thres:
                word_count.pop(word, None)
        
        word_list = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        word_list += list(word_count.keys())
        
        return {word:i for i, word in enumerate(word_list)}, [word for i, word in enumerate(word_list)]

    def build_relation_dict(self, relation_list_path):
        with open(relation_list_path) as f:
            tmp = f.read().split("\n")

        if tmp[-1] == "":
            tmp = tmp[:-1]

        relation_list = []

        for i in tmp:
            if i[0] != "#":
                relation_list.append(i)
 
        return {relation[4:]:i for i, relation in enumerate(relation_list)}

    def batch_sentence_to_index(self, sentence):
        with mp.Pool() as p:
            sentence_index = list(p.map(self.sentence_to_index, sentence))
            
        return sentence_index

    def sentence_to_index(self, sentence):
        tmp = [self.vocab["<SOS>"]]
        for word in sentence.split(" "):
            if word in self.vocab.keys():
                tmp.append(self.vocab[word])
            else:
                tmp.append(self.vocab["<UNK>"])
        tmp.append(self.vocab["<EOS>"])
        return tmp

    def batch_position_to_index(self, position):
        with mp.Pool() as p:
            position_index = list(p.map(self.position_to_index, position))
            
        return position_index

    def position_to_index(self, position):
        tmp_position = [position[0] - 1] + position + [position[-1] + 1]
        
        tmp = []
        for i in tmp_position:
            tmp.append(max(0, min((self.position_window_size * 2), self.position_window_size + i)))

        return tmp

    def batch_relation_to_index(self, relation):
        with mp.Pool() as p:
            relation_index = list(p.map(self.relation_to_index, relation))
            
        return relation_index

    def relation_to_index(self, relation):
        return self.relation_dict[relation]

    def batch_relation_to_onehot(self, relation):
        with mp.Pool() as p:
            relation_onehot = list(p.map(self.relation_to_onehot, relation))
            
        return relation_onehot

    def relation_to_onehot(self, relation):
        tmp = [0] * len(self.relation_dict.keys())
        if relation in self.relation_dict.keys():
            tmp[self.relation_dict[relation]] = 1

            return tmp
        
        return [1] * len(self.relation_dict.keys())

    def batch_no_relation_to_onehot(self, relation):
        with mp.Pool() as p:
            relation_onehot = list(p.map(self.no_relation_to_onehot, relation))
            
        return relation_onehot

    def no_relation_to_onehot(self, relation):
        if relation not in self.relation_dict.keys():
            return [0, 1]
        else:
            return [-1, 0]

    def padding_sentence_index(self, index, max_len = None):
        if max_len is None:
            max_len = max([len(i) for i in index])
        
        tmp = []
        for i in index:
            if len(i) >  max_len:
                tmp.append(i[:max_len])
            else:
                tmp.append(i + ([self.vocab['<PAD>']] * (max_len - len(i))))
                
        return tmp
    
    def padding_position_index(self, index, max_len = None):
        if max_len is None:
            max_len = max([len(i) for i in index])
        
        tmp = []
        for i in index:
            if len(i) >  max_len:
                tmp.append(i[:max_len])
            else:
                tmp.append(i + ([(self.position_window_size * 2) + 1] * (max_len - len(i))))
                
        return tmp

    def batch_forward(self, sentence, entity_position, filler_position, batch_size):
        tmp = list(range(len(sentence)))

        sentence = np.array(sentence)
        entity_position = np.array(entity_position)
        filler_position = np.array(filler_position)

        for i in range(0, len(sentence), batch_size):
            yield sentence[tmp[i:i + batch_size]].tolist(), entity_position[tmp[i:i + batch_size]].tolist(), filler_position[tmp[i:i + batch_size]].tolist()
            
    def batch_train(self, sentence, entity_position, filler_position, relation, label, batch_size, random):
        tmp = list(range(len(sentence)))
            
        if random == True:
            np.random.shuffle(tmp)
        
        sentence = np.array(sentence)
        entity_position = np.array(entity_position)
        filler_position = np.array(filler_position)
        relation = np.array(relation)
        label = np.array(label)
        
        for i in range(0, len(sentence), batch_size):
            yield sentence[tmp[i:i + batch_size]].tolist(), entity_position[tmp[i:i + batch_size]].tolist(), filler_position[tmp[i:i + batch_size]].tolist(), relation[tmp[i:i + batch_size]].tolist(), label[tmp[i:i + batch_size]].tolist()
         
    def length_to_onehot(self, length, max_len):
        tmp_list = []
        for i in length:
            tmp = [0] * max_len
            if i > max_len:
                tmp[-1] = 1
            else:
                tmp[i - 1] = 1
            tmp_list.append(tmp)
            
        return tmp_list
        
    def data_to_onehot(self, data):
        with mp.Pool() as p:
            data_onehot = list(p.map(self.sentence_to_onehot, data))
            
        return data_onehot
        
    def index_to_data(self, index):
        with mp.Pool() as p:
            sentence = list(p.map(self.index_to_sentence, index))
            
        return sentence
    
    def padding_onehot(self, onehot, max_len = None):
        if max_len is None:
            max_len = max([len(i) for i in onehot])
        
        tmp = []
        for i in onehot:
            if len(i) >  max_len:
                tmp.append(i[:max_len])
            else:
                tmp.append(i + ([[0] * len(self.vocab.keys())] * (max_len - len(i))))
                
        return tmp
    
    def sentence_to_onehot(self, sentence):
        tmp = [0] * len(self.vocab.keys())
        tmp[self.vocab["<SOS>"]] = 1
        
        tmp_list = [tmp]
        
        for word in sentence.split(" "):
            tmp = [0] * len(self.vocab.keys())
            if word in self.vocab.keys():
                tmp[self.vocab[word]] = 1
            else:
                tmp[self.vocab["<UNK>"]] = 1
                
            tmp_list.append(tmp)
        
        tmp = [0] * len(self.vocab.keys())
        tmp[self.vocab["<EOS>"]] = 1
        tmp_list.append(tmp)
            
        return tmp_list
    
    def index_to_sentence(self, index):
        tmp = "<SOS>"
        for i in index:
            tmp += " "
            tmp += self.reverse_vocab[i]
            
        return tmp




