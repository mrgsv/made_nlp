# BoW

from collections import defaultdict
vocab = defaultdict(int)
for text in texts_train:
    splitted_text = text.split()
    for string in splitted_text:
        vocab[string] += 1

def text_to_bow(text, vocabulary):
    """ convert text string to an array of token counts. Use bow_vocabulary. """
    output = np.zeros(len(vocabulary), dtype="float32")
    for word in text.split():
        try:
            output[vocabulary.index(word)] += 1
        except ValueError as e:
            continue
    return output

# AUC for different size of dictionary

train_aucs, test_aucs = [], []
for k in range(1000, 5001, 1000):
    bow_vocabulary_k = list({k: v for k, v in sorted(vocab.items(), key=lambda x: x[1], reverse=True)}.keys())[:k]
    X_train_bow = np.stack(list(map(lambda x: text_to_bow(text=x, vocabulary=bow_vocabulary_k), texts_train)))
    X_test_bow = np.stack(list(map(lambda x: text_to_bow(text=x, vocabulary=bow_vocabulary_k), texts_test)))
    X_train_bow_torch = torch.FloatTensor(X_train_bow)
    X_test_bow_torch = torch.FloatTensor(X_test_bow)
    model_with_k = nn.Sequential()
    model_with_k.add_module("l1", nn.Linear(in_features=k, out_features=2))
    opt = torch.optim.Adam(model_with_k.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(optimizer=opt)
    train_model(model_with_k, opt, lr_scheduler, X_train_bow_torch, y_train_torch, X_test_bow_torch, y_test_torch)
    for name, X, y, model_with_k in [
        ('train', X_train_bow_torch, y_train, model_with_k),
        ('test ', X_test_bow_torch, y_test, model_with_k)
    ]:
        proba = model_with_k(X).detach().cpu().numpy()[:, 1]
        auc = roc_auc_score(y, proba)
        if name == 'train':
            train_aucs.append(auc)
        else:
            test_aucs.append(auc)
        plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

    plt.plot([0, 1], [0, 1], '--', color='black',)
    plt.legend(fontsize='large')
    plt.grid()

# TF-IDF
compliance = {word: idx for idx, word in enumerate(vocab.keys())}

from collections import defaultdict
def get_tf_idf(text, word_to_number, alpha=1):
    tf_map = np.empty((0, len(word_to_number)), dtype="float32")
    word_occurences = np.zeros(len(word_to_number), dtype="int32")
    word_occurences_dict = defaultdict(set)
    for idx, line in enumerate(text):
        tf_map_cur = np.zeros(len(word_to_number))
        splitted_line = line.split()
        for word in splitted_line:
            try:
                tf_map_cur[word_to_number[word]] += 1
                word_occurences_dict[word].add(idx)
            except KeyError as e:
                continue
        tf_map_cur = np.array(list(map(lambda x: x / len(tf_map_cur), tf_map_cur)))
        tf_map = np.vstack((tf_map, tf_map_cur))
    for word in word_occurences_dict.keys():
        try:
            word_occurences[word_to_number[word]] = len(word_occurences_dict[word])
        except KeyError as e:
            continue
    idf_scores = np.array(list(map(lambda cnt: np.log(len(text) / (cnt + alpha)), word_occurences)))
    return tf_map * idf_scores, tf_map, idf_scores

# TF-IDF model
tf_idf_model = nn.Sequential()
tf_idf_model.add_module("l1", nn.Linear(in_features=X_train_tfidf_torch.shape[1], out_features=2))
loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(tf_idf_model.parameters(), lr=1e-3)
lr_scheduler = ReduceLROnPlateau(optimizer=opt, cooldown=10)

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
multinomialNB_bow = MultinomialNB()
multinomialNB_tf_idf = MultinomialNB()

multinomialNB_bow.fit(X=X_train_bow, y=y_train)
multinomialNB_tf_idf.fit(X=tf_idf_map_train_normalized, y=y_train)

# W2V
import gensim.downloader as api
vec_size = 25
gensim_model = api.load('glove-twitter-' + str(vec_size))

new_vocab = ["<UNK>", "<PAD>"]
for word in vocab:
    try:
        gensim_model.get_vector(word)
        new_vocab.append(word)
    except KeyError as e:
        continue
new_vocab = {word: idx for idx, word in enumerate(new_vocab)}

UNK_IX, PAD_IX = map(new_vocab.get, ["<UNK>", "<PAD>"])

def as_matrix(sequences, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))
        
    max_len = min(max(map(len, sequences)), max_len or float('inf'))
    
    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
    for i,seq in enumerate(sequences):
        row_ix = [new_vocab.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix
    
    return matrix

word2vec_emb_matrix = np.random.normal(size=(2, vec_size))
for word in new_vocab:
    try:
        word2vec_emb_matrix = np.vstack((word2vec_emb_matrix, gensim_model.get_vector(word)))
    except KeyError as e:
        continue
        
word2vec_emb_matrix = torch.FloatTensor(word2vec_emb_matrix)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))

n_tokens = len(new_vocab)
hid_size = vec_size
n_maximums = 3

simple_model = nn.Sequential()

simple_model.add_module('emb', nn.Embedding.from_pretrained(embeddings=word2vec_emb_matrix, padding_idx=PAD_IX, freeze=False))
simple_model.add_module('reorder', Reorder())
simple_model.add_module('conv1', nn.Conv1d(
    in_channels=hid_size,
    out_channels=hid_size,
    kernel_size=2
))
simple_model.add_module('bn1', nn.BatchNorm1d(hid_size))
simple_model.add_module('adaptive_pool', nn.AdaptiveMaxPool1d(output_size=n_maximums))
simple_model.add_module('relu1', nn.ReLU())
simple_model.add_module('flatten', nn.Flatten())
simple_model.add_module('out', nn.Linear(in_features=hid_size*n_maximums, out_features=2))