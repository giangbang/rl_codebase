import numpy as np


class RunningMeanDict(dict):
    def update(self, dict_: dict):
        for key, val in dict_.items():
            if key not in self:
                self[key] = []
            self[key].append(val)
    
    def to_dict(self):
        res = {}
        for key, val in self.items():
            res[key] = np.mean(val)
        return res