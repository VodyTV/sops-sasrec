# main
import argparse
import os
import json

import numpy as np
import tensorflow as tf
from sas_rec.dataset import data_partition, SasRecSequence
from sas_rec.models.model import SASRecModel
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


class Experiment(App):
    """
    Experiment class
    """
    def load_dataset(self):

        fname = self.args.f_name
        self.dataset = data_partition(fname)
        [user_train, user_valid, user_test, user_num, item_num] = self.dataset
        self.item_num = item_num
        self.user_num = user_num

        if self.args.shuffle_sequence:
            print(self.args.shuffle_sequence)

        train_sequence = SasRecSequence(
            user_train,
            self.user_num, self.item_num,
            batch_size=self.args.batch_size,
            max_sequence_len=self.args.maxlen,
            shuffle_seq=self.args.shuffle_sequence,
            seed=self.args.seed
        )
        return train_sequence

    def train_model(self, train_sequence):
        """
        Train up the model
        """

        self.model = SASRecModel(
            self.item_num,
            maxlen=self.args.maxlen,
            hidden_dim=self.args.hidden_dim,
            num_heads=self.args.num_heads,
            num_blocks=self.args.num_blocks,
            dropout=self.args.dropout_rate
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_2=0.98)
        self.model.compile(
            optimizer
        )

        # epoch_end = EvalCallback(self.dataset, every_n=self.args.num_epochs)

        history = self.model.fit(
            train_sequence,
            epochs=self.args.num_epochs,
            # callbacks=[epoch_end],
            verbose=2
        )
        return history

    def evaluate_model(self):
        """
        Evaluate metrics on trained model
        """
        val_ndcg, val_hr, val_ranks = self.model.evaluate_valid(self.dataset)
        test_ndcg, test_hr, test_ranks = self.model.evaluate_test(self.dataset)
        outs = {
            "Val-NDCG_at_10": val_ndcg,
            "Val-HR_at_10": val_hr,
            "Test-NDCG_at_10": test_ndcg,
            "Test-HR_at_10": test_hr,
        }
        return outs

    def run(self):
        print("Loading Dataset...")
        tf.random.set_seed(self.args.seed)
        train_sequence = self.load_dataset()
        print("train model...")
        history = self.train_model(train_sequence)
        print(history.history)
        print("evaluate model...")
        metrics = self.evaluate_model()
        metrics["final_loss"] = np.round(history.history['loss'][-1], 4)
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
