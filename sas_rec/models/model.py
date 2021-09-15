# Model
import random

import numpy as np
import tensorflow as tf


class SASFeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 num_units=[2048, 512],
                 dropout_rate=0.2,
                 name="SASFeedForward",
                 **kwargs):
        super(SASFeedForward, self).__init__(name=name, **kwargs)
        self.num_units = num_units
        self.dropout_rate = dropout_rate

        # Layers
        self.convone = tf.keras.layers.Conv1D(
            self.num_units[0], kernel_size=1, activation='relu')
        self.dropout_one = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.convtwo = tf.keras.layers.Conv1D(self.num_units[1], kernel_size=1)
        self.dropout_two = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.add_layer = tf.keras.layers.Add()

    def call(self, inputs, training):
        # Inner layer
        outputs = self.convone(inputs)
        outputs = self.dropout_one(outputs, training=training)
        # Readout layer
        outputs = self.convtwo(outputs)
        outputs = self.dropout_two(outputs, training=training)
        # Residual connection
        outputs = self.add_layer([inputs, outputs])
        return outputs


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_dim,
                 num_heads=2,
                 num_blocks=1,
                 dropout_rate=0.2,
                 name="AttentionBlock",
                 training=True,
                 **kwargs):
        """
        Attention Block
        """
        super(AttentionBlock, self).__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.training = training

        # Layers
        self.mhattention = tf.keras.layers.MultiHeadAttention(
            num_heads,
            key_dim=hidden_dim,
            value_dim=None,
            dropout=self.dropout_rate,
            use_bias=True,
            output_shape=None,
            attention_axes=(1),  # attn on sequence [batch, seq, emb]
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
        )
        self.feedforward = SASFeedForward(num_units=[self.hidden_dim, self.hidden_dim],
                                          dropout_rate=self.dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, attn_mask, seq_mask, training=True):
        x_normed, x = inputs
        # Self-attention
        x = self.mhattention(query=x_normed,
                             value=x,
                             key=x,
                             training=training,
                             attention_mask=attn_mask,
                             )
        # Residual Connection
        x = tf.keras.layers.Add()([x, x_normed])
        x = self.layernorm(x)

        # Feed Forward
        x = self.feedforward(x, training)
        x = tf.keras.layers.Multiply()([seq_mask, x])
        return x


class SASRecModel(tf.keras.Model):
    """
    Classic SASRec model
    """
    def __init__(
        self,
        item_num,
        maxlen,
        hidden_dim,
        num_heads,
        num_blocks,
        dropout,
        l2_reg=0.
    ):
        super(SASRecModel, self).__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.l2_reg = l2_reg

        # Masking Layer
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.)

        # Sequence Embedding
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=(self.item_num + 1),
            output_dim=(self.hidden_dim),
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=self.l2_reg)
        )

        self.positional_embedding = tf.keras.layers.Embedding(
            # An embedding for each position in our sequence
            input_dim=(self.maxlen),
            output_dim=(self.hidden_dim),  # Embeddings have this dimensionality,
            embeddings_regularizer=tf.keras.regularizers.l2(l2=self.l2_reg)
        )

        self.dropout_one = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.attention_block = AttentionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate)

        self.add_layer = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.last_layer_norm = tf.keras.layers.LayerNormalization()
        self.pos_multiply = tf.keras.layers.Multiply()
        self.neg_multiply = tf.keras.layers.Multiply()

        self.reducesum_layer1 = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-1))
        self.reducesum_layer2 = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-1))

        self.test_dot = tf.keras.layers.Dot(axes=(1, 2))
        self.reducesum_test = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=-1))

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.pos_loss_tracker = tf.keras.metrics.Mean(name="pos_loss")
        self.neg_loss_tracker = tf.keras.metrics.Mean(name="neg_loss")

    def log2feats(self, sequence, positions, training=True):
        # Get sequence emebeddings and scale
        x = self.item_embedding(sequence) * (self.hidden_dim ** 0.5)

        # Get positional encoding
        positional_encoding = self.positional_embedding(positions)

        # Make sequence mask tensor
        sequence_mask = tf.expand_dims(
            tf.cast(tf.not_equal(sequence, 0), dtype=tf.float32), -1)

        x = self.add_layer([x, positional_encoding])
        x = self.dropout_one(x, training=training)
        x = tf.keras.layers.Multiply()([sequence_mask, x])

        # Attention Mask to enforce causality
        attention_mask = tf.cast(
            tf.cast(tf.experimental.numpy.tril(tf.ones((self.maxlen, self.maxlen))),
                    dtype=tf.bool), dtype=tf.int32)

        x_normed = self.layer_norm(x)

        for i in range(self.num_blocks):
            x = self.attention_block(
                [x_normed, x], attention_mask, sequence_mask, training=training)

        x = self.last_layer_norm(x)
        return x

    def call(self, inputs, training=True):
        # Forward
        user, seq, pos, neg, positions = inputs

        x = self.log2feats(seq, positions, training=training)

        pos_emb = self.item_embedding(pos) * (self.hidden_dim ** 0.5)
        neg_emb = self.item_embedding(neg) * (self.hidden_dim ** 0.5)

        pos_muls = self.pos_multiply([pos_emb, x])
        neg_muls = self.neg_multiply([neg_emb, x])

        pos_logits = self.reducesum_layer1(pos_muls)
        neg_logits = self.reducesum_layer2(neg_muls)

        return (pos_logits, neg_logits)

    def train_step(self, data):
        x, y = data
        pos_labels, neg_labels = y
        pos = x[2]
        istarget = tf.cast(tf.not_equal(pos, 0), dtype=tf.float32)

        pos_labels = pos_labels * istarget
        neg_labels = neg_labels * istarget

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            pos_logits, neg_logits = self(x, training=True)  # Forward pass

            # Compute the loss value
            pos_loss = bce(y_true=pos_labels, y_pred=pos_logits * istarget)
            neg_loss = bce(y_true=neg_labels, y_pred=neg_logits * istarget)
            loss = pos_loss + neg_loss + sum(self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.pos_loss_tracker.update_state(pos_loss)
        self.neg_loss_tracker.update_state(neg_loss)

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(),
                "pos_loss": self.pos_loss_tracker.result(),
                "neg_loss": self.neg_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def predict(self, seq, item_indicies):

        # position indicies of input sequence
        positions = tf.expand_dims(np.arange(self.maxlen), axis=0)

        # Get logits for input sequence
        log_feats = self.log2feats(seq, positions, training=False)
        final_features = tf.reshape(log_feats[:, -1, :], shape=(1, 50))
        item_embeds = self.item_embedding(
            tf.convert_to_tensor([item_indicies])) * (self.hidden_dim ** 0.5)

        test_logits = self.test_dot([final_features, item_embeds])

        return test_logits

    def evaluate_valid(self, dataset):
        """
        Evaluate on validation item ([ seq ], [ val ], [ test ])
        """
        [train, valid, test, usernum, itemnum] = dataset
        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0
        # If dataset has more than 10K Users take a sample of 10K for the metrics
        if usernum > 10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = range(1, usernum + 1)

        ranks = []
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            # User History
            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            test_logits = self.predict(tf.convert_to_tensor([seq]), item_idx)
            predictions = -test_logits[0]
            rank = predictions.numpy().argsort().argsort()[0]
            ranks.append(rank)
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                print('.', end=' ')

        return np.round(NDCG / valid_user, 4), np.round(HT / valid_user, 4), ranks

    def evaluate_test(self, dataset):
        """
        Evaluate on test item ([ seq, val ], [ test ])
        """
        [train, valid, test, usernum, itemnum] = dataset
        NDCG = 0.0
        HT = 0.0
        valid_user = 0.0

        # If dataset has more than 10K Users take a sample of 10K for the metrics

        if usernum > 10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = range(1, usernum + 1)

        ranks = []
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            # Append validation item to sequence
            seq[idx] = valid[u][0]
            # Work back along original sequence
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
            # User History
            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated:
                    t = np.random.randint(1, itemnum + 1)
                item_idx.append(t)

            test_logits = self.predict(tf.convert_to_tensor([seq]), item_idx)
            predictions = -test_logits[0]

            rank = predictions.numpy().argsort().argsort()[0]
            ranks.append(rank)
            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                print('.', end=' ')

        return np.round(NDCG / valid_user, 4), np.round(HT / valid_user, 4), ranks
