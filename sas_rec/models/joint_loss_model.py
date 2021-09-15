# Joint Loss Model

import tensorflow as tf


class SASFeedForward(tf.keras.layers.Layer):
    def __init__(self,
                 num_units=[2048, 512],
                 dropout_rate=0.2,
                 l2_reg=0.,
                 name="SASFeedForward",
                 **kwargs):
        super(SASFeedForward, self).__init__(name=name, **kwargs)
        self.num_units = num_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Layers
        self.convone = tf.keras.layers.Conv1D(
            self.num_units[0], kernel_size=1, activation='relu',
            activity_regularizer=tf.keras.regularizers.l2(l2=self.l2_reg))
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
                 l2_reg=0.,
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
        self.l2_reg = l2_reg
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
            activity_regularizer=tf.keras.regularizers.l2(l2=self.l2_reg)
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


class SASRecJLModel(tf.keras.Model):
    def __init__(
        self,
        item_num,
        maxlen,
        hidden_dim,
        num_heads,
        num_blocks,
        dropout,
        class_weights,
        l2_reg=0.
    ):
        super(SASRecJLModel, self).__init__()
        self.item_num = item_num
        self.maxlen = maxlen
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.class_weights = class_weights
        self.l2_reg = l2_reg

        # Masking Layer
        self.masking_layer = tf.keras.layers.Masking(mask_value=0.)

        # Sequence Embedding
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=(self.item_num + 1),
            output_dim=(self.hidden_dim),
            mask_zero=True,
        )

        self.positional_embedding = tf.keras.layers.Embedding(
            # An embedding for each position in our sequence
            input_dim=(self.maxlen),
            output_dim=(self.hidden_dim),  # Embeddings have this dimensionality,
        )

        self.dropout_one = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.attention_block = AttentionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg
        )

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

        self.dense_out = tf.keras.layers.Dense(6, activation='softmax')

        # Trackings
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
        self.cce_metric = tf.keras.metrics.CategoricalCrossentropy(name="cce")
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
        # shape [batch, maxlen, hidden_dim]

        x = self.dense_out(x)
        # shape [batch, maxlen, 6]

        return x

    def call(self, inputs, training=True):
        # Forward
        user, seq, positions = inputs
        x = self.log2feats(seq, positions, training=training)
        # Return predicted scores for items
        return x

    def train_step(self, data):
        x, y, _ = data
        y_true = y

        sw = tf.reshape([self.class_weights[i] for i in range(0, 6)], shape=(1, 6))

        # istarget = tf.expand_dims(tf.cast(tf.not_equal(x[1], 0), dtype=tf.float32), -1)
        non_zero_sequence_elements = tf.cast(tf.not_equal(x[1], 0), dtype=tf.float32)

        scaled_nzse = tf.reshape(
            tf.tensordot(sw, y, axes=((1), (2))),
            non_zero_sequence_elements.shape
        )

        # Categorical Cross entropy
        mse = tf.keras.losses.MeanSquaredError()
        cce = tf.keras.losses.CategoricalCrossentropy()

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # shape: (batch_size, maxlen, 6)
            # sample weight says which element of the sequences are zero
            loss = cce(y_true, y_pred, sample_weight=scaled_nzse * non_zero_sequence_elements)

            # Contribution to loss that accounts for closesness of rating prediction
            true_ratings = tf.cast(tf.argmax(y_true, axis=-1), 'float32')
            pred_ratings = tf.cast(tf.argmax(y_pred, axis=-1), 'float32')

            loss += mse(non_zero_sequence_elements * true_ratings,
                        non_zero_sequence_elements * pred_ratings)
            loss += sum(self.losses)

        # Compute gradients cat
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(non_zero_sequence_elements * true_ratings,
                                     non_zero_sequence_elements * pred_ratings)
        self.cce_metric.update_state(y_true, y_pred,
                                     sample_weight=scaled_nzse * non_zero_sequence_elements)
        self.acc_metric.update_state(non_zero_sequence_elements * true_ratings,
                                     non_zero_sequence_elements * pred_ratings)

        # Return a dict mapping metric names to current value
        return {
            "loss": self.loss_tracker.result(),
            "mse": self.mse_metric.result(),
            "cce": self.cce_metric.result(),
            "acc": self.acc_metric.result()
        }

    @property
    def metrics(self):
        """
        We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`.
        If you don't implement this property, you have to call
        `reset_states()` yourself at the time of your choosing.
        """
        return [self.loss_tracker, self.mse_metric, self.cce_metric, self.acc_metric]

    def test_step(self, data):
        # Unpack the data
        x, y, sw = data
        # Compute predictions
        y_pred = self(x, training=False)

        # Just the last n
        y_pred = y_pred[:, -1:]
        y_true = y[:, -1:]

        # Turn the classification pick into the rating value
        true_ratings = tf.cast(tf.argmax(y_true, axis=-1), 'float32')
        pred_ratings = tf.cast(tf.argmax(y_pred, axis=-1), 'float32')

        mse = tf.keras.losses.MeanSquaredError()
        cce = tf.keras.losses.CategoricalCrossentropy()

        cce_loss = cce(y_true, y_pred)
        mse_loss = mse(true_ratings, pred_ratings)
        loss = cce_loss + mse_loss

        # UPDATE LOSS
        self.loss_tracker.update_state(loss)

        # Update metrics
        self.mse_metric.update_state(true_ratings, pred_ratings)
        self.cce_metric.update_state(y_true, y_pred)
        self.acc_metric.update_state(true_ratings, pred_ratings)

        return {
            "loss": self.loss_tracker.result(),
            "mse": self.mse_metric.result(),
            "cce": self.cce_metric.result(),
            "acc": self.acc_metric.result(),
        }
