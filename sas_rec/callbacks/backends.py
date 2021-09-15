import json
import os


METRIC_BACKENDS = {}


class MetricBackend:
    def __init__(self, args):
        self.args = args

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        # register subclass
        name = getattr(cls, 'backend_name', cls.__name__)
        if name not in METRIC_BACKENDS:
            METRIC_BACKENDS[name] = cls

    @staticmethod
    def lookup_backend(name):
        if name not in METRIC_BACKENDS:
            all_backends = ', '.join(METRIC_BACKENDS.keys())
            raise ValueError(f'Metric backend `{name}` not found in [{all_backends}]')
        return METRIC_BACKENDS.get(name)


class EchoMetricBackend(MetricBackend):
    backend_name = 'echo'

    def __call__(self, metrics):
        print(metrics)


class KubeflowMetricBackend(MetricBackend):
    backend_name = 'kubeflow'

    def __call__(self, metrics):
        output_metrics = dict(
            metrics=[
                dict(name=key_to_kf_metric_name(key), numberValue=float(value))
                for key, value in metrics.items()
            ]
        )

        os.makedirs(os.path.dirname(self.args.output_metrics_path), exist_ok=True)
        with open(self.args.output_metrics_path, 'w') as outfile:
            json.dump(output_metrics, outfile)


def key_to_kf_metric_name(k):
    k = k.replace('_', '-').lower()
    return k
