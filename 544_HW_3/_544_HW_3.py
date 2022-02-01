import pandas as pd
import numpy as np
from sklearn import metrics
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import json 

def read_vocab(df_train):
    vocab = df_train.drop(columns=['Index', 'Pos'], axis=1)
    vocab = vocab.groupby(['Value'], sort=False).size().reset_index(name='Count')
    vocab = vocab.groupby(['Value'], as_index=False).sum()
    vocab = vocab.sort_values('Count', ascending=False)
    vocab['Index'] = np.arange(vocab.shape[0])

    cols = list(vocab.columns)
    a, b = cols.index('Count'), cols.index('Index')
    cols[b], cols[a] = cols[a], cols[b]
    vocab = vocab[cols]

    print("The size of the vocabulary is: ", len(vocab))
    print("There are 0 unknown words as no unk threshold is used")

    vocab.to_csv("vocab.txt", sep="\t", header=False, index=False)
    return vocab

def build_dicts(train_pos, df_pos_val, pos_d):
    trans_dict, em_dict = {}, {}

    for i in range(len(train_pos)-1):
        if train_pos[i][0] == 1:
            if '--s--' + train_pos[i + 1][1] not in trans_dict:
                trans_dict['--s--' + train_pos[i + 1][1]] = 1/38360
            else:
                trans_dict['--s--' + train_pos[i + 1][1]] += 1/38360
        else:
            if train_pos[i][1] + train_pos[i + 1][1] not in trans_dict:
                trans_dict[train_pos[i][1] + train_pos[i + 1][1]] = 1/pos_d[train_pos[i][1]]
            else:
                trans_dict[train_pos[i][1] + train_pos[i + 1][1]] += 1/pos_d[train_pos[i][1]]

    for i,r in df_pos_val.iterrows():
        em_dict[r['Pos'] + r['Value']] = r['Count']/pos_d[r['Pos']]

    return trans_dict, em_dict

def greedy_predict(x_dev, hmm, pos_values):
    predicted = []
    for i in range(len(x_dev)):
        greedy_prob = float('-inf');
        pos_pred = ''
        if x_dev[i][1] in vd:
            for j in range(len(pos_values)):
                if x_dev[i][0] == 1:
                    pos_pair = '--s--' + pos_values[j]
                else:
                    pos_pair = predicted[i-1] + pos_values[j]

                if pos_pair not in hmm['transition']:
                    continue

                t_prob = hmm['transition'][pos_pair]
                em_pair = pos_values[j] + x_dev[i][1]

                if em_pair not in hmm['emission']:
                    continue

                em_prob = hmm['emission'][em_pair]
                temp = np.log(em_prob) + np.log(t_prob)
                if temp > greedy_prob:
                    greedy_prob = temp
                    pos_pred = pos_values[j]

        if pos_pred == '':
            for j in range(len(pos_values)):
                t_prob = np.nextafter(0,1)

                if x_dev[i][0] == 1:
                    pos_pair = '--s--' + pos_values[j]
                else:
                    pos_pair = predicted[i-1] + pos_values[j]

                if pos_pair not in hmm['transition']:
                    continue

                t_prob = hmm['transition'][pos_pair]

                temp = np.log(t_prob)
                if temp > greedy_prob:
                    greedy_prob = temp
                    pos_pred = pos_values[j]

        predicted.append(pos_pred)

    return predicted

def get_sentences(df):
    sent =[]
    corpus = []
    for i,r in df.iterrows():
        if r['Index'] == 1:
            corpus.append(sent)
            sent = []
        sent.append(r['Value'])
    corpus.append(sent)
    corpus.pop(0)
    return corpus

def map_word(x, vocab_idx):
    if x in vocab_idx:
        return vocab_idx[x]
    else:
        return -1

def calc_T_matrix(trans_dict, pos_values):
    t_matrix = [[None for i in range(len(pos_values))] for j in range(len(pos_values))] 
    for i in range(len(pos_values)):
        for j in range(len(pos_values)):
            trans_pair = pos_values[i]+pos_values[j]
            if trans_pair in trans_dict:
                t_matrix[i][j] = trans_dict[trans_pair]
            else:
                t_matrix[i][j] = np.nextafter(0, 1)
    return t_matrix  

def calc_E_matrix(em_dict, pos_values, vocab):
    e_matrix = [[None for i in range(len(vocab))] for j in range(len(pos_values))] 
    for i in range(len(vocab)):
        for j in range(len(pos_values)):
            em_pair = pos_values[j]+vocab[i]
            if em_pair in em_dict:
                e_matrix[j][i] = em_dict[em_pair]
            else:
                e_matrix[j][i] = np.nextafter(0, 1)
    return e_matrix  

def viterbi_initialize(test, em_dict, trans_dict, pos_values):
    best_probs = []
    best_paths = []
    for sent in test:
        sent_probs = [[0 for i in range(len(sent))] for j in range(len(pos_values))] 
        sent_path = [[0 for i in range(len(sent))] for j in range(len(pos_values))] 
        for j in range(len(pos_values)):
            bestProbToWordIFromTagJ = float('-inf')
            bestPathToWordI = 0
            trans_prob = np.nextafter(0, 1)
            em_prob = np.nextafter(0, 1)
            trans_pair = '--s--' + pos_values[j]
            em_pair = pos_values[j]+sent[0]

            if trans_pair in trans_dict:
                trans_prob = trans_dict[trans_pair]
            if em_pair in em_dict:
                em_prob = em_dict[em_pair]

            sent_probs[j][0] = np.log(trans_prob) + np.log(em_prob)
            sent_path[j][0] = j
        best_probs.append(sent_probs)
        best_paths.append(sent_path)

    return best_probs, best_paths

def viterbi_forward(test, em_dict, trans_dict, pos_values, best_probs, best_paths, t_matrix, e_matrix):
    for x in range(len(test)):     
        for i in range(1, len(test[x])):
            for j in range(len(pos_values)):
                bestProbToWordIFromTagJ = float('-inf')
                bestPathToWordI = 0
                for k in range(len(pos_values)):
                    trans_prob = t_matrix[k][j]
                    em_prob = np.nextafter(0, 1)
                    
                    if test[x][i] > -1:
                        em_prob=e_matrix[j][test[x][i]]
                    
                    temp_prob = best_probs[x][k][i-1] + np.log(trans_prob) + np.log(em_prob)

                    if (temp_prob > bestProbToWordIFromTagJ):
                        bestProbToWordIFromTagJ = temp_prob
                        bestPathToWordI = k

                best_probs[x][j][i] = bestProbToWordIFromTagJ
                best_paths[x][j][i] = bestPathToWordI

    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, pos_values):
    predicted = []
    for i in range(len(best_probs)):
        m = len(best_probs[i][0])
        z = [None] * m
        bestProbForLastWord = float('-inf')
        pred_pos = [None] * m
        for k in range(len(pos_values)): 
            temp = best_probs[i][k][m - 1]
            if temp > bestProbForLastWord:
                bestProbForLastWord = temp
                z[m - 1] = k
        pred_pos[m-1] = pos_values[z[m-1]]
        for j in reversed(range(m)): 
            if j != 0:
                tagForWordI = best_paths[i][z[j]][j]
                z[j - 1] = tagForWordI
                pred_pos[j-1] = pos_values[tagForWordI]
        predicted += pred_pos
    return predicted

df_train = pd.read_csv("data/train", sep = '\t', names=["Index", "Value", "Pos"])
df_train.loc[df_train.Value.str.contains(r'^(?:[+-]|\()?\$?\d+(?:,\d+)*(?:\.\d+)?\)?$'), 'Value'] = '--num--'
vocab =  read_vocab(df_train)

vd = pd.Series(vocab.Count.values,index=vocab.Value).to_dict()
vl = vocab['Value'].to_numpy()
vd_idx = pd.Series(vocab.Index.values,index=vocab.Value).to_dict()

df_pos = df_train.groupby(['Pos'], sort=False).size().reset_index(name='Count')
pos_d = pd.Series(df_pos.Count.values,index=df_pos.Pos).to_dict()
pos_values = pd.Series(df_pos.Pos.values)

train_pos = df_train[['Index', 'Pos']].to_numpy()
df_pos_val = df_train.groupby(['Value','Pos'], sort=False).size().reset_index(name='Count')

trans_dict, em_dict = build_dicts(train_pos, df_pos_val, pos_d)

print("Number of transition parameters: ", len(trans_dict), " Number of emission parameters: ", len(em_dict))

temp = {}
temp['transition'] = trans_dict
temp['emission'] = em_dict

with open('hmm.json', 'w') as outfile:
    json.dump(temp, outfile)

f = open('hmm.json')
hmm = json.load(f)

df_dev = pd.read_csv("data/dev", sep = '\t', names=["Index", "Value", "Pos"])
df_dev.loc[df_dev.Value.str.contains(r'^(?:[+-]|\()?\$?\d+(?:,\d+)*(?:\.\d+)?\)?$'), 'Value'] = '--num--'
x_dev, y_dev  = df_dev[['Index','Value']].to_numpy(), df_dev['Pos'].to_numpy()

g_pred_dev = greedy_predict(x_dev, hmm, pos_values)

print("Greedy HMM Dev Accuracy: ", metrics.accuracy_score(y_dev, g_pred_dev)) 

df_test = pd.read_csv("data/test", sep = '\t', names=["Index", "Value"])
temp = pd.read_csv("data/test", sep = '\t', names=["Index", "Value"])

temp.loc[temp.Value.str.contains(r'^(?:[+-]|\()?\$?\d+(?:,\d+)*(?:\.\d+)?\)?$'), 'Value'] = '--num--'
df_test_greedy = df_test_viterbi = temp

x_test = df_test_greedy[['Index','Value']].to_numpy()

g_pred_test = greedy_predict(x_test, hmm, pos_values)
df_test_greedy['Predicted'] = g_pred_test

output = pd.DataFrame(columns=["Index", "Value", "Predicted"])
for i, row in df_test.iterrows():
    row['Predicted'] = g_pred_test[i]
    if row['Index'] == 1 and i != 0:
        temp = pd.DataFrame([[None, None, None]], columns=["Index", "Value", "Predicted"])
        temp = temp.append(row)
        output = output.append(temp, ignore_index=True)
    else:
        output = output.append(row)

output.to_csv('greedy.out', sep="\t", header=False, index=False)   

t_matrix = calc_T_matrix(hmm['transition'], pos_values)
e_matrix = calc_E_matrix(hmm['emission'], pos_values, vl)

v_map_word = np.vectorize(map_word)

corpus=get_sentences(df_dev)
corpus_mapped =[]
for sent in corpus:
    corpus_mapped.append(v_map_word(sent, vd_idx))

best_probs, best_paths = viterbi_initialize(corpus, hmm['emission'], hmm['transition'], pos_values)
best_probs_final, best_paths_final = viterbi_forward(corpus_mapped, hmm['emission'], hmm['transition'], pos_values, best_probs, best_paths, t_matrix, e_matrix)
v_pred_dev = viterbi_backward(best_probs_final, best_paths_final, pos_values)

print("Viterbi HMM Accuracy: ", metrics.accuracy_score(y_dev, v_pred_dev))  

corpus_test=get_sentences(df_test_viterbi)
corpus_mapped_test =[]
for sent in corpus_test:
    corpus_mapped_test.append(v_map_word(sent, vd_idx))

best_probs_test, best_paths_test = viterbi_initialize(corpus_test, hmm['emission'], hmm['transition'], pos_values)
best_probs_final_test, best_paths_final_test = viterbi_forward(corpus_mapped_test, hmm['emission'], hmm['transition'], pos_values, best_probs_test, best_paths_test, t_matrix, e_matrix)
v_pred_test = viterbi_backward(best_probs_final_test, best_paths_final_test, pos_values)

df_test_viterbi['Predicted'] = v_pred_test

output = pd.DataFrame(columns=["Index", "Value", "Predicted"])
for i, row in df_test.iterrows():
    row['Predicted'] = v_pred_test[i]
    if row['Index'] == 1 and i != 0:
        temp = pd.DataFrame([[None, None, None]], columns=["Index", "Value", "Predicted"])
        temp = temp.append(row)
        output = output.append(temp, ignore_index=True)
    else:
        output = output.append(row)

output.to_csv('viterbi.out', sep="\t", header=False, index=False)   