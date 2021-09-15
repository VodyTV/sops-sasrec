# main
import argparse
from collections import Counter
import os
import json

import numpy as np
import tensorflow as tf
from sas_rec.callbacks.callbacks import DropClassWeightsCallback
from sas_rec.dataset import data_partition_ratings, SasRecJLSequence
from sas_rec.models.joint_loss_model import SASRecJLModel
from sas_rec.app import App


def str2bool(variable):
    if isinstance(variable, bool):
        return variable
    if variable.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif variable.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calc_class_weights(ratings):

    probs = {}
    for k, v in Counter(ratings).items():
        probs[int(k)] = round(v / sum(Counter(ratings).values()), 4)

    inverse_prob_sum = sum([1 / p for p in probs.values()])

    class_weights = {}
    for k, v in probs.items():
        class_weights[int(k)] = round(5 / (inverse_prob_sum * v), 4)
    class_weights[0] = 0.0
    return class_weights


class Experiment(App):
    """
    Experiment class
    """
    def load_dataset(self):
        """
        Load dataset from file and return train, val and test sequences
        """

        fname = self.args.f_name
        assert 'rating' in fname

        print("Shuffled?\t", self.args.shuffle_sequence)

        self.holdout_n = 1
        print("Holdout N: ", self.holdout_n)

        self.dataset = data_partition_ratings(fname,
                                              seed=self.args.seed,
                                              holdout_n=self.holdout_n,
                                              shuffle=self.args.shuffle_sequence)
        [user_train, user_valid, user_test, user_num, item_num, uir] = self.dataset
        self.item_num = item_num
        self.user_num = user_num
        self.uir = uir  # User - Item _Rating dict

        ratings = []
        for i in range(1, self.user_num + 1):
            for thing in list(self.uir[i].values()):
                ratings.append(thing)

        self.class_weights = calc_class_weights(ratings)
        print("Class Weights:", self.class_weights)

        train_sequence = SasRecJLSequence(users=user_train,
                                          uir_map=uir,
                                          num_users=self.user_num,
                                          num_items=self.item_num,
                                          batch_size=self.args.batch_size,
                                          max_sequence_len=self.args.maxlen,
                                          seed=self.args.seed,
                                          class_weights=self.class_weights,
                                          train=True)

        train_user_order = []
        for tsq in train_sequence:
            train_user_order += list(tsq[0][0])

        new_user_valid = {}
        for user in train_user_order:
            new_val_seq = user_train[user] + user_valid[user]
            new_user_valid[user] = new_val_seq

        validation_sequence = SasRecJLSequence(
            users=new_user_valid,
            uir_map=uir,
            num_users=self.user_num,
            num_items=self.item_num,
            batch_size=self.args.batch_size,
            max_sequence_len=self.args.maxlen,
            seed=self.args.seed,
            train=False,
            class_weights=self.class_weights,
            user_order=train_user_order)

        new_user_test = {}
        for user in train_user_order:
            new_test_seq = user_train[user] + user_valid[user] + user_test[user]
            new_user_test[user] = new_test_seq

        test_sequence = SasRecJLSequence(
            users=new_user_test,
            uir_map=uir,
            num_users=self.user_num,
            num_items=self.item_num,
            batch_size=self.args.batch_size,
            max_sequence_len=self.args.maxlen,
            seed=self.args.seed,
            train=False,
            class_weights=self.class_weights,
            user_order=train_user_order)

        return train_sequence, validation_sequence, test_sequence

    def train_model(self, train_sequence, validation_sequence):
        """
        Train up the model
        """

        self.model = SASRecJLModel(
            self.item_num,
            maxlen=self.args.maxlen,
            hidden_dim=self.args.hidden_dim,
            num_heads=self.args.num_heads,
            num_blocks=self.args.num_blocks,
            dropout=self.args.dropout_rate,
            class_weights=self.class_weights,
            l2_reg=self.args.l2_reg,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr, beta_2=0.98)
        self.model.compile(
            optimizer,
            sample_weight_mode='temporal',
            run_eagerly=True
        )

        history = self.model.fit(
            train_sequence,
            epochs=self.args.num_epochs,
            verbose=1,
            validation_data=validation_sequence,
            callbacks=[DropClassWeightsCallback(when_step=int(self.args.num_epochs / 2))]
        )
        return history

    def evaluate_model(self, test_sequence):
        """
        Evaluate metrics on trained model
        """
        result = self.model.evaluate(test_sequence)
        # outs = {
        #     'test_mse': mse,
        #     'test_cce': cce,
        #     'test_acc': acc

        # }

        outs = dict(zip(self.model.metrics_names, result))
        outs = {"test_" + k: np.round(v, 4) for k, v in outs.items()}
        return outs

    def run(self):
        print("Loading Dataset...")
        tf.random.set_seed(self.args.seed)
        train_sequence, validation_sequence, test_sequence = self.load_dataset()
        print("train model...")
        history = self.train_model(train_sequence, validation_sequence)
        print(history.history)
        print("evaluate model...")
        metrics = self.evaluate_model(test_sequence)
        metrics["val_mse"] = np.round(history.history['val_mse'][-1], 4)
        metrics["val_cce"] = np.round(history.history['val_cce'][-1], 4)
        metrics["val_acc"] = np.round(history.history['val_acc'][-1], 4)
        metrics["final_loss"] = np.round(history.history['loss'][-1], 4)
        metrics["best_loss"] = np.round(min(history.history['loss']), 4)
        self.send_metrics(metrics)
        return metrics

    @classmethod
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--f-name', type=str, default="ml-1m")
        parser.add_argument("--shuffle-sequence", type=str2bool, nargs='?',
                            const=True, default=False, help="Activate nice mode.")
        parser.add_argument('--batch-size', default=128, type=int)
        parser.add_argument('--lr', default=0.001, type=float)
        parser.add_argument('--maxlen', default=200, type=int)
        parser.add_argument('--hidden-dim', default=50, type=int)
        parser.add_argument('--num-blocks', default=2, type=int)
        parser.add_argument('--num-epochs', default=201, type=int)
        parser.add_argument('--num-heads', default=1, type=int)
        parser.add_argument('--dropout-rate', default=0.2, type=float)
        parser.add_argument('--l2-emb', default=0.0, type=float)
        parser.add_argument('--l2-reg', default=0.2, type=float)
        parser.add_argument('--seed', default=101, type=int)
        parser.add_argument('--metric-backend', default='kubeflow')
        parser.add_argument('--output-metrics-path',
                            help='Location of output metrics.')

    def key_to_kf_metric_name(self, k):
        k = k.replace('_', '-').lower()
        return k

    def send_metrics(self, metrics):
        output_metrics = dict(
            metrics=[
                dict(name=self.key_to_kf_metric_name(key), numberValue=float(value))
                for key, value in metrics.items()
            ]
        )
        print(output_metrics)

        os.makedirs(os.path.dirname(self.args.output_metrics_path), exist_ok=True)
        with open(self.args.output_metrics_path, 'w') as outfile:
            json.dump(output_metrics, outfile)


if __name__ == '__main__':
    print("GO...")
    experiment = Experiment.create_from_args()
    experiment.run()
