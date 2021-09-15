# Model

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


class SASRecRegModel(tf.keras.Model):
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
        super(SASRecRegModel, self).__init__()
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

        self.dense_out = tf.keras.layers.Dense(1)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.acc_metric = tf.keras.metrics.Accuracy(name="acc")

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
        x = self.dense_out(x)  # could be five units (one for each score) - classification?
        return x

    def call(self, inputs, training=True):
        # Forward
        user, seq, positions = inputs
        x = self.log2feats(seq, positions, training=training)
        # Return predicted scores for items
        return x

    def train_step(self, data):
        x, y = data
        y_true = y
        istarget = tf.expand_dims(tf.cast(tf.not_equal(x[1], 0), dtype=tf.float32), -1)
        mse = tf.keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Compute the loss value
            mse_loss = mse(y_true, y_pred * istarget)
            loss = mse_loss + sum(self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(y, y_pred)
        self.mae_metric.update_state(y, y_pred)
        # Round predictions to nearest int
        self.acc_metric.update_state(y, tf.round(y_pred))

        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(),
                "mse": self.mse_metric.result(),
                "mae": self.mae_metric.result(),
                "acc": self.acc_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_metric, self.mae_metric, self.acc_metric]

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)

        # Just want the last two
        y_pred = y_pred[:, -2:]
        y_true = y[:, -2:]

        self.mse_metric.update_state(y_true, y_pred)
        self.mae_metric.update_state(y_true, y_pred)
        # Round predictions to nearest int
        self.acc_metric.update_state(y_true, tf.round(y_pred))
        return {"mse": self.mse_metric.result(),
                "mae": self.mae_metric.result(),
                "acc": self.acc_metric.result()}
