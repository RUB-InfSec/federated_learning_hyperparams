import ray


@ray.remote
class ClientCommunicator:
    def __init__(self):
        self.active_client = -1,
        self.active_clients = {
            'benign': [],
            'malicious': [],
        }
        self.something = {}

    def reset(self):
        self.active_client = -1,
        self.active_clients = {
            'benign': [],
            'malicious': [],
        }

    def set_malicious(self, cids):
        self.active_clients['malicious'] = cids

    def set_benign(self, cids):
        self.active_clients['benign'] = cids

    def set_active_id(self, cid):
        self.active_client = cid

    def get_active_id(self):
        return self.active_client

    def get_total(self):
        return len(self.active_clients['malicious']) + len(self.active_clients['benign'])

    def get_malicious(self):
        return len(self.active_clients['malicious'])

    def get_list_of_attackers(self):
        return self.active_clients['malicious']

    def store_something(self, key, value):
        self.something[key] = value

    def get_something(self, key):
        return self.something.get(key)