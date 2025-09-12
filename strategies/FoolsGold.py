from typing import Optional, List, Tuple, Dict, Union, Callable

import numpy as np
from flwr.common import Parameters, FitRes, parameters_to_ndarrays, ndarrays_to_parameters, Scalar, NDArrays, \
    MetricsAggregationFn
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import sklearn.metrics.pairwise as smp

epsilon = 1e-5

'''
Returns the pairwise cosine similarity of client gradients
'''


def get_cos_similarity(full_deltas):
    return smp.cosine_similarity(full_deltas)


def importanceFeatureMapGlobal(model):
    return np.abs(model) / np.sum(np.abs(model))


def flatten_arrays(array_list):
    """
    Flatten a list of numpy arrays into a single numpy array.

    Parameters:
    array_list (list of numpy.ndarray): List of numpy arrays to be flattened.

    Returns:
    tuple: A tuple containing the flattened numpy array and a list of shapes of the original arrays.
    """
    shapes = [array.shape for array in array_list]
    flattened_array = np.concatenate([array.flatten() for array in array_list])
    return flattened_array, shapes


def restore_arrays(flattened_array, shapes):
    """
    Restore a flattened numpy array back into a list of arrays with their original shapes.

    Parameters:
    flattened_array (numpy.ndarray): The flattened numpy array.
    shapes (list of tuple): List of shapes of the original arrays.

    Returns:
    list of numpy.ndarray: List of numpy arrays restored to their original shapes.
    """
    restored_arrays = []
    start_idx = 0
    for shape in shapes:
        size = np.prod(shape)
        array = flattened_array[start_idx:start_idx + size].reshape(shape)
        restored_arrays.append(array)
        start_idx += size
    return restored_arrays


class FoolsGold(FedAvg):
    """FoolsGold aggregation strategy as defined in "Defending against Backdoors in Federated Learning with Robust Learning Rate" (Fung et al., RAID, 2020)

    Implementation based on https://github.com/DistributedML/FoolsGold

    Parameters
    ----------
    ... (cf. flwr.server.strategy.fedavg.FedAvg)
    importance: str (default: "")
        The strategy to selecte most important features. Accepted values are "hard" and "map_local"
    topk: float (default: 0.05)
        The portion of output layer features to consider being indicative features.
    memory_size: int (default: 0)
        The number of previous rounds' updates to store and compute cosine similarity on for each client.
    """

    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            num_clients_per_round: int = None,
            num_clients: int = None,
            memory_size: int = 0,
            importance: str = "",
            topk: float = 0.05,
    ) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures, initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=False)
        self.current_weight = parameters_to_ndarrays(initial_parameters)
        self.num_features = len(flatten_arrays(self.current_weight)[0])
        self.num_classes = len(self.current_weight[-1])
        self.num_clients_per_round = num_clients_per_round
        self.num_clients = num_clients
        self.importance = importance
        self.topk = topk
        self.memory_size = memory_size
        self.delta_memory = np.zeros((self.num_clients, self.num_features, self.memory_size))
        self.summed_deltas = np.zeros((self.num_clients_per_round, self.num_features))

    def __repr__(self) -> str:
        return "FoolsGold"

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        flattened_current_weights, _ = flatten_arrays(self.current_weight)

        sig_features_idx = np.arange(self.num_features)

        updates_per_client = [
            [w_new - w for w, w_new in zip(self.current_weight, parameters_to_ndarrays(result[1].parameters))]
            for result in results]

        # Cope for failures
        self.num_clients_per_round = len(updates_per_client)
        self.summed_deltas = np.zeros((self.num_clients_per_round, self.num_features))

        # Count total examples
        num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

        # Compute scaling factors for each result
        scaling_factors = [
            fit_res.num_examples / num_examples_total for _, fit_res in results
        ]

        flattened_updates_per_client, shape = [], None
        for i, update in enumerate(updates_per_client):
            flattened_update, shape = flatten_arrays(update)

            # Rescale update by client's dataset size
            flattened_update *= scaling_factors[i]

            flattened_updates_per_client.append(flattened_update)
        flattened_updates_per_client = np.array(flattened_updates_per_client)

        if self.memory_size > 0:
            partition_ids = []
            for flattened_update, result in zip(flattened_updates_per_client, results):
                partition_id = result[0].partition_id
                partition_ids.append(partition_id)

                # normalize delta
                if np.linalg.norm(flattened_update) > 1:
                    flattened_update /= np.linalg.norm(flattened_update)

                self.delta_memory[partition_id, :, server_round % self.memory_size] = flattened_update

            # Track the total vector from each individual client
            self.summed_deltas = np.sum(self.delta_memory[np.array(partition_ids)], axis=2)
        else:
            for flattened_update in flattened_updates_per_client:
                # normalize delta
                if np.linalg.norm(flattened_update) > 1:
                    flattened_update /= np.linalg.norm(flattened_update)

            # Track the total vector from each individual client
            self.summed_deltas += flattened_updates_per_client

        flattened_current_weights += self.foolsgold(flattened_updates_per_client, self.summed_deltas, sig_features_idx)

        self.current_weight = restore_arrays(flattened_current_weights, shape)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(self.current_weight), metrics_aggregated

    def importanceFeatureMapLocal(self, flattened_current_weights):
        # According to the original paper, the feature importance filtering is only applied to the output layer of
        # neural networks In the given setting, the weights of the output layer are stored in the second-to-last
        # entry of the weights list
        num_features_output_layer = self.current_weight[-2].shape[1]

        M = flattened_current_weights[-(num_features_output_layer + 1) * self.num_classes:-self.num_classes].copy()

        M = np.reshape(M, (self.num_classes, num_features_output_layer))

        for i in range(self.num_classes):
            M[i] = np.abs(M[i] - M[i].mean())

            M[i] = M[i] / M[i].sum()

            # Top k of 784
            topk = int(num_features_output_layer * self.topk)
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            M[i][sig_features_idx] = 0

        return M.flatten()

    def importanceFeatureHard(self, flattened_current_weights):
        # According to the original paper, the feature importance filtering is only applied to the output layer of
        # neural networks In the given setting, the weights of the output layer are stored in the second-to-last
        # entry of the weights list
        num_features_output_layer = self.current_weight[-2].shape[1]

        M = flattened_current_weights[-(num_features_output_layer + 1) * self.num_classes:-self.num_classes].copy()
        M = np.reshape(M, (self.num_classes, num_features_output_layer))

        importantFeatures = np.ones((self.num_classes, num_features_output_layer))
        # Top k of 784
        topk = int(num_features_output_layer * self.topk)
        for i in range(self.num_classes):
            sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
            importantFeatures[i][sig_features_idx] = 0

        return importantFeatures.flatten()

    def foolsgold(self, this_delta, summed_deltas, sig_features_idx):
        # Take all the features of sig_features_idx for each client
        sd = summed_deltas.copy()
        sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

        if self.importance in ["map_local", "hard"]:
            flattened_current_weights, _ = flatten_arrays(self.current_weight)
            if self.importance == "map_local":
                # smooth version of importance features
                importantFeatures = self.importanceFeatureMapLocal(flattened_current_weights)
            if self.importance == "hard":
                # hard version of important features
                importantFeatures = self.importanceFeatureHard(flattened_current_weights)
            for i in range(self.num_clients_per_round):
                num_features_output_layer = self.current_weight[-2].shape[1]
                sig_filtered_deltas[i][
                -(num_features_output_layer + 1) * self.num_classes:-self.num_classes] *= importantFeatures

        if np.isnan(sig_filtered_deltas).any():
            print("[WARN] NaNs detected in client updates before cosine similarity.")
        sig_filtered_deltas = np.nan_to_num(sig_filtered_deltas, nan=0.0)

        cs = smp.cosine_similarity(sig_filtered_deltas) - np.eye(self.num_clients_per_round)
        # Pardoning: reweight by the max value seen
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(self.num_clients_per_round):
            for j in range(self.num_clients_per_round):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        wv = 1 - (np.max(cs, axis=1))
        wv[wv > 1] = 1
        wv[wv < 0] = 0

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # Apply the weight vector on this delta
        delta = np.reshape(this_delta, (self.num_clients_per_round, self.num_features))

        return np.dot(delta.T, wv)
