import tensorflow as tf


class EvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, every_n=20):
        self.dataset = dataset
        self.every_n = every_n

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        if epoch % self.every_n == 0:
            val_ndcg, val_hr, val_ranks = self.model.evaluate_valid(self.dataset)
            print("\nEnd epoch {} of training; got log keys: {}".format(epoch, keys))
            print("VALIDATION")
            print("VAL - NDCG@10: ", val_ndcg)
            print("VAL - HR@10:", val_hr)
            test_ndcg, test_hr, test_ranks = self.model.evaluate_test(self.dataset)
            print("TEST")
            print("TEST - NDCG@10: ", test_ndcg)
            print("TEST - HR@10:", test_hr)
            print("\n")


class DropClassWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, when_step=10):
        super(DropClassWeightsCallback, self).__init__()
        self.when_step = when_step

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.when_step - 1:
            self.model.class_weights = {k: 1.0 if k != 0 else 0 for k, v
                                        in self.model.class_weights.items()}
