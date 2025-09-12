# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bulyan [El Mhamdi et al., 2018] strategy.

Paper: arxiv.org/abs/1802.07927
"""


from logging import WARNING
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate_krum, _find_reference_weights, aggregate_median, \
    _aggregate_n_closest_weights
from flwr.server.strategy.fedavg import FedAvg

def aggregate_bulyan(
        results: List[Tuple[NDArrays, int]],
        num_malicious: int,
        aggregation_rule: Callable,  # type: ignore
        **aggregation_rule_kwargs: Any,
) -> NDArrays:
    """Perform Bulyan aggregation.

    Parameters
    ----------
    results: List[Tuple[NDArrays, int]]
        Weights and number of samples for each of the client.
    num_malicious: int
        The maximum number of malicious clients.
    aggregation_rule: Callable
        Byzantine resilient aggregation rule used as the first step of the Bulyan
    aggregation_rule_kwargs: Any
        The arguments to the aggregation rule.

    Returns
    -------
    aggregated_parameters: NDArrays
        Aggregated parameters according to the Bulyan strategy.
    """
    byzantine_resilient_single_ret_model_aggregation = [aggregate_krum]
    # also GeoMed (but not implemented yet)
    byzantine_resilient_many_return_models_aggregation = []  # type: ignore
    # Brute, Medoid (but not implemented yet)

    num_clients = len(results)
    if num_clients < 4 * num_malicious + 3:
        log(WARNING,
            "The Bulyan aggregation requires then number of clients to be greater or "
            "equal to the 4 * num_malicious + 3. This is the assumption of this method."
            "It is needed to ensure that the method reduces the attacker's leeway to "
            "the one proved in the paper."
        )
    selected_models_set: List[Tuple[NDArrays, int]] = []

    theta = len(results) - 2 * num_malicious
    beta = theta - 2 * num_malicious

    if beta <= 0:
        raise ValueError(
            f"There are not enough local updates ({len(results)}) in this round to perform Bulyan aggregation."
            f"This causes beta to become {beta}."
        )

    for _ in range(theta):
        best_model = aggregation_rule(
            results=results, num_malicious=num_malicious, **aggregation_rule_kwargs
        )
        list_of_weights = [weights for weights, num_samples in results]
        # This group gives exact result
        if aggregation_rule in byzantine_resilient_single_ret_model_aggregation:
            best_idx = _find_reference_weights(best_model, list_of_weights)
        # This group requires finding the closest model to the returned one
        # (weights distance wise)
        elif aggregation_rule in byzantine_resilient_many_return_models_aggregation:
            # when different aggregation strategies available
            # write a function to find the closest model
            raise NotImplementedError(
                "aggregate_bulyan currently does not support the aggregation rules that"
                " return many models as results. "
                "Such aggregation rules are currently not available in Flower."
            )
        else:
            raise ValueError(
                "The given aggregation rule is not added as Byzantine resilient. "
                "Please choose from Byzantine resilient rules."
            )

        selected_models_set.append(results[best_idx])

        # remove idx from tracker and weights_results
        results.pop(best_idx)

    # Compute median parameter vector across selected_models_set
    median_vect = aggregate_median(selected_models_set)

    # Take the averaged beta parameters of the closest distance to the median
    # (coordinate-wise)
    parameters_aggregated = _aggregate_n_closest_weights(
        median_vect, selected_models_set, beta_closest=beta
    )
    return parameters_aggregated


def contains_nan(ndarrays: NDArrays) -> bool:
    return any(np.isnan(arr).any() for arr in ndarrays)


# flake8: noqa: E501
# pylint: disable=line-too-long
class Bulyan(FedAvg):
    """Bulyan strategy.

    Implementation based on https://arxiv.org/abs/1802.07927.

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    num_malicious_clients : int, optional
        Number of malicious clients in the system. Defaults to 0.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    first_aggregation_rule: Callable
        Byzantine resilient aggregation rule that is used as the first step of the Bulyan (e.g., Krum)
    **aggregation_rule_kwargs: Any
        arguments to the first_aggregation rule
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            num_malicious_clients: int = 0,
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
            first_aggregation_rule: Callable = aggregate_krum,  # type: ignore
            **aggregation_rule_kwargs: Any,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.num_malicious_clients = num_malicious_clients
        self.first_aggregation_rule = first_aggregation_rule
        self.aggregation_rule_kwargs = aggregation_rule_kwargs

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"Bulyan(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using Bulyan."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Filter out models that contain NaN values
        weights_results = [
            (weights, num_samples)
            for (weights, num_samples) in weights_results
            if not contains_nan(weights)
        ]

        # Aggregate weights
        parameters_aggregated = ndarrays_to_parameters(
            aggregate_bulyan(
                weights_results,
                self.num_malicious_clients,
                self.first_aggregation_rule,
                **self.aggregation_rule_kwargs,
            )
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
