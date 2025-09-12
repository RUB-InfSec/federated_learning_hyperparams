import ray
from flwr.server.client_manager import SimpleClientManager
from flwr.server.criterion import Criterion
from flwr.server.client_proxy import ClientProxy
from flwr.common import logger
from logging import INFO
from typing import Dict, List, Optional
import random

class CustomClientManager(SimpleClientManager):
    def __init__(self, poisoning_scheduler):
        super().__init__()
        self.poisoning_scheduler = poisoning_scheduler
        self.client_communicator = ray.get_actor("client_communicator", namespace="clients")

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            logger.log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)

        # Ensure that client with partition_id 0 (the attacker client) is present in poisoning rounds for fixed-frequency poisoning
        partition_ids_to_cids = {str(v.partition_id): k for k, v in self.clients.items()}
        cids_to_partition_ids = {k: str(v.partition_id) for k, v in self.clients.items()}

        if self.poisoning_scheduler.type == 'fixed_frequency' and self.poisoning_scheduler.poison("0"):
            # smaller partition ids belong to attackers
            for i in range(self.poisoning_scheduler.num_malicious_clients):
                index = None
                if partition_ids_to_cids[str(i)] not in sampled_cids:
                    # replace first benign client in the list
                    index = next(
                        (idx for idx, cid in enumerate(sampled_cids) if not self.poisoning_scheduler.poison(cids_to_partition_ids[cid]))
                    )
                if index is not None:
                    sampled_cids[index] = partition_ids_to_cids[str(i)]


        elif self.poisoning_scheduler.type == 'interval_fixed':
            available_malicious_cids = [cid for cid in available_cids if self.poisoning_scheduler.poison(cids_to_partition_ids[cid])]
            available_benign_cids = [cid for cid in available_cids if not self.poisoning_scheduler.poison(cids_to_partition_ids[cid])]
            if len(available_malicious_cids) > 0:
                sampled_cids = random.sample(available_malicious_cids, self.poisoning_scheduler.malicious_clients_per_round) + random.sample(available_benign_cids, num_clients - self.poisoning_scheduler.malicious_clients_per_round)
        elif self.poisoning_scheduler.type == 'round_wise':
            available_malicious_cids = [cid for cid in available_cids if self.poisoning_scheduler.poison(cids_to_partition_ids[cid])]
            available_benign_cids = [cid for cid in available_cids if not self.poisoning_scheduler.poison(cids_to_partition_ids[cid])]
            if len(available_malicious_cids) > 0:
                sampled_cids = available_malicious_cids + random.sample(available_benign_cids, num_clients - len(available_malicious_cids))

        malicious_cids = [cids_to_partition_ids[cid] for cid in sampled_cids if self.poisoning_scheduler.poison(cids_to_partition_ids[cid])]
        self.client_communicator.set_malicious.remote(malicious_cids)
        if len(malicious_cids) > 0:
            self.client_communicator.set_active_id.remote(malicious_cids[0])
        benign_cids = [cids_to_partition_ids[cid] for cid in sampled_cids if cids_to_partition_ids[cid] not in malicious_cids]
        self.client_communicator.set_benign.remote(benign_cids)

        return [self.clients[cid] for cid in sampled_cids]