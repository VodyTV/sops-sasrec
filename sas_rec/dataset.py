# Dataset
from collections import defaultdict
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import cast, tile
from tensorflow import expand_dims
from tensorflow.keras.utils import Sequence


def data_partition(fname):
    """
    Author function for loading and arranging datasets
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def data_partition_ratings(fname, seed=101, holdout_n=1, shuffle=False):
    """
    Author function for loading and arranging datasets
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_item_rating = {}
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i, r = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        if i == 0:
            print(line)
        r = float(r)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
        if u not in user_item_rating:
            user_item_rating[u] = {}
        user_item_rating[u][i] = r

    user_dict_items = list(User.items())
    np.random.seed(seed)
    np.random.shuffle(user_dict_items)
    User = dict(user_dict_items)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            if shuffle:
                user_train[user] = list(np.random.permutation(User[user][:-holdout_n * 2]))
            else:
                user_train[user] = User[user][:-holdout_n * 2]
            user_valid[user] = []
            user_valid[user].extend(User[user][-holdout_n * 2:-holdout_n])
            user_test[user] = []
            user_test[user].extend(User[user][-holdout_n:])
    return [user_train, user_valid, user_test, usernum, itemnum, user_item_rating]


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class SasRecSequence(Sequence):

    def __init__(self, users, num_users, num_items, batch_size, max_sequence_len,
                 shuffle_seq=False, seed=101):
        self.users = users
        self.num_users = num_users
        self.num_items = num_items
        self.user_ids = list(self.users.keys())
        self.batch_size = batch_size
        self.max_len = max_sequence_len
        self.shuffle_seq = shuffle_seq
        self.seed = seed
        np.random.seed(self.seed)
        print("Is shuffled?\n\t", self.shuffle_seq)

    def __len__(self):
        return math.ceil(len(self.user_ids) / self.batch_size)

    def __getitem__(self, idx):
        batch_users = []
        user_sequences = []  # [self.users[user] for user in batch_users]
        positives = []
        negatives = []
        nexts = []

        for _ in range(self.batch_size):

            user = np.random.randint(1, self.num_users + 1)  # select a random user

            # if the user has no sequence, select another
            while len(self.users[user]) <= 1:
                user = np.random.randint(1, self.num_users + 1)

            batch_users.append(user)

            user_seq = np.zeros([self.max_len], dtype=np.int32)
            pos = np.zeros([self.max_len], dtype=np.int32)
            neg = np.zeros([self.max_len], dtype=np.int32)
            nxt = self.users[user][-1]
            idx = self.max_len - 1

            # Go backwards through sequence
            user_history = set(self.users[user])
            for i in reversed(self.users[user][:-1]):
                user_seq[idx] = i
                pos[idx] = nxt
                if nxt != 0:
                    neg[idx] = random_neq(1, self.num_items + 1, user_history)
                nxt = i
                idx -= 1
                if idx == -1:
                    break
            if self.shuffle_seq:
                nonzero_idx = np.nonzero(user_seq)
                user_seq[nonzero_idx] = np.random.permutation(user_seq[nonzero_idx])
            user_sequences.append(user_seq)
            positives.append(pos)
            negatives.append(neg)
            nexts.append(nxt)

        positions = cast(tile(x=expand_dims(np.arange(self.max_len), axis=0),
                              n=[len(batch_users), 1]), "int32").numpy()

        # return
        # x : batch_users, user_sequences, positive_targets, negative_targets, positions
        # y : positive labels, negative labels
        return (
            np.array(batch_users, dtype='int32'), np.array(user_sequences),
            np.array(positives), np.array(negatives), positions), (
            np.ones((len(batch_users), self.max_len)), np.zeros((len(batch_users), self.max_len)))


class SasRecModSequence(Sequence):
    """
    Sets up sasrec for the modified training paradigm

    X is the same
    Y is last n items with their ratings
    """
    def __init__(self, users, uir_map, num_users, num_items, batch_size, max_sequence_len,
                 seed=101):
        self.users = users
        self.uir = uir_map
        self.num_users = num_users
        self.num_items = num_items
        self.user_ids = list(self.users.keys())
        self.batch_size = batch_size
        self.max_len = max_sequence_len

    def __len__(self):
        return math.ceil(len(self.user_ids) / self.batch_size)

    def __getitem__(self, idx):
        batch_users = []
        user_sequences = []  # [self.users[user] for user in batch_users]
        user_ratings = []

        batch_users = self.user_ids[idx * self.batch_size:(idx + 1) * self.batch_size]

        for user in batch_users:
            user_seq_items = np.zeros([self.max_len], dtype=np.int32)
            user_seq_ratings = np.zeros([self.max_len], dtype=np.int32)
            idx = self.max_len - 1

            # Go backwards through sequence
            for i in reversed(self.users[user]):
                user_seq_items[idx] = i
                idx -= 1
                if idx == -1:
                    break
            self.uir[user][0] = 0
            user_sequences.append(user_seq_items)
            user_seq_ratings = [self.uir[user][item] for item in user_seq_items]
            user_ratings.append(user_seq_ratings)

        positions = cast(tile(x=expand_dims(np.arange(self.max_len), axis=0),
                              n=[len(batch_users), 1]), "int32").numpy()

        # return
        # x : batch_users, user_sequences, ratings
        # y : positive labels, negative labels
        return (np.array(batch_users, dtype='int32'),
                np.array(user_sequences), positions), (np.array(user_ratings))


class SasRecJLSequence(Sequence):
    """
    Sets up sasrec for the modified training paradigm

    X is the same
    Y is last n items with their ratings
    """
    def __init__(self, users, uir_map, num_users, num_items,
                 batch_size, max_sequence_len,
                 class_weights,
                 num_batches=None,
                 train=True,
                 seed=101,
                 user_order=None):
        self.input_users = users
        self.uir = uir_map
        self.num_users = num_users
        self.num_items = num_items
        self.user_ids = list(self.input_users.keys())

        self.batch_size = batch_size
        self.max_len = max_sequence_len
        self.class_weights = class_weights

        self.train = train
        self.seed = seed
        self.user_order = user_order

        if not num_batches:
            self.num_batches = math.ceil(len(self.user_ids) / self.batch_size)
        else:
            self.num_batches = num_batches

        np.random.seed(self.seed)
        self.generated_users = np.random.randint(
            1, self.num_users + 1, size=(self.num_batches, self.batch_size))

    def __len__(self):
        return math.ceil(len(self.user_ids) / self.batch_size)

    def __getitem__(self, idx):
        batch_users = []
        user_sequences = []
        user_ratings = []

        for jdx in range(self.batch_size):
            if self.train:
                user = self.generated_users[idx][jdx]
                batch_users.append(user)
            else:
                user = self.user_order[idx * self.batch_size + jdx]
                batch_users.append(user)

            user_seq_items = np.zeros([self.max_len], dtype=np.int32)
            user_seq_ratings = np.zeros([self.max_len], dtype=np.int32)
            seq_idx = self.max_len - 1

            # Go backwards through sequence
            for i in reversed(self.input_users[user]):
                user_seq_items[seq_idx] = i
                seq_idx -= 1
                if seq_idx == -1:
                    break
            # Define value to give for item index 0
            self.uir[user][0] = 0
            user_sequences.append(user_seq_items)
            user_seq_ratings = [self.uir[user][item] for item in user_seq_items]
            user_ratings.append(user_seq_ratings)

        positions = cast(tile(x=expand_dims(np.arange(self.max_len), axis=0),
                              n=[len(batch_users), 1]), "int32").numpy()

        sample_weights = tf.reshape([self.class_weights[i] for i in range(0, 6)], shape=(1, 6))

        # return
        # x : batch_users, user_sequences, ratings
        # y : one-hot ratings
        return ((np.array(batch_users, dtype='int32'), np.array(user_sequences), positions),
                (to_categorical(user_ratings, num_classes=6)),
                (sample_weights))
