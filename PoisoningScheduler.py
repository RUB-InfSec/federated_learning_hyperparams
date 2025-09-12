from abc import abstractmethod

class PoisoningScheduler:
    def __init__(self, num_malicious_clients: int, type: str = 'interval'):
        self.num_malicious_clients = num_malicious_clients
        self.current_round = 0
        self.type = type

    @abstractmethod
    def poison(self, cid: str) -> bool:
        pass

    @abstractmethod
    def get_last_poisoning_round(self, num_rounds: int) -> int:
        pass

    @abstractmethod
    def get_first_poisoning_round(self) -> int:
        pass


class IntervalPoisoningScheduler(PoisoningScheduler):
    def __init__(self, num_malicious_clients, start_round: int, end_round: int = None, type: str = 'interval'):
        super().__init__(num_malicious_clients, type=type)
        self.start_round = start_round
        self.end_round = end_round

    def poison(self, cid: str) -> bool:
        if int(cid) >= self.num_malicious_clients:
            return False
        if self.end_round:
            return self.start_round <= self.current_round <= self.end_round
        else:
            return self.current_round >= self.start_round

    def get_last_poisoning_round(self, num_rounds: int) -> int:
        return self.end_round if self.end_round else num_rounds

    def get_first_poisoning_round(self) -> int:
        return self.start_round

class IntervalFixedPoisoningScheduler(IntervalPoisoningScheduler):
    def __init__(self, num_malicious_clients, start_round: int, malicious_clients_per_round: int, end_round: int = None):
        super().__init__(num_malicious_clients, start_round, end_round, 'interval_fixed')
        self.malicious_clients_per_round = malicious_clients_per_round


class FrequencyPoisoningScheduler(PoisoningScheduler):
    def __init__(self, start_round: int, end_round: int = None, frequency: int = None, malicious_clients_per_round: int = 1):
        super().__init__(malicious_clients_per_round, 'fixed_frequency')
        self.start_round = start_round
        self.end_round = end_round
        self.frequency = frequency

    def poison(self, cid: str) -> bool:
        if int(cid) >= self.num_malicious_clients:
            return False
        if (not self.end_round and self.start_round <= self.current_round) or (
                self.start_round <= self.current_round <= self.end_round):
            return (self.current_round - self.start_round) % self.frequency == 0
        else:
            return False

    def get_last_poisoning_round(self, num_rounds: int) -> int:
        return self.end_round if self.end_round else num_rounds

    def get_first_poisoning_round(self) -> int:
        return self.start_round

class RoundWisePoisoningScheduler(PoisoningScheduler):
    def __init__(self, malicious_clients_dict):
        malicious_cids = []
        for round, malicious_cids in malicious_clients_dict.items():
            for cid in malicious_cids:
                if cid not in malicious_cids:
                    malicious_cids.append(cid)
        super().__init__(len(malicious_cids), 'round_wise')
        self.malicious_clients_dict = malicious_clients_dict

    def poison(self, cid: str) -> bool:
        if self.current_round in self.malicious_clients_dict.keys():
            return int(cid) in self.malicious_clients_dict[self.current_round]
        else:
            return False

    def get_first_poisoning_round(self) -> int:
        return min(self.malicious_clients_dict.keys())

    def get_last_poisoning_round(self, num_rounds: int) -> int:
        return max(self.malicious_clients_dict.keys())
