import csv
from rl_codebase.core.utils import get_time_now_as_str


class Logger:
    """
    Logger class, logs output to stdout and csv file
    The structure of the saved folder is as follows:
    `log_dir` 
    │
    └───`env_name`
    │   │
    │   └───`exp_name`
    │       │   `file_name`
    │       │   `file_name`
    │       │   ...
    │   ...
    """
    def __init__(self, log_dir='logs', env_name='env_name', exp_name='exp_name',
            file_name='progress.csv'):
        self.log_dir = log_dir
        self.log_to_file = log_dir is not None
        self.name_to_vals = {}
        
        if self.log_to_file:
            if not file_name.endswith('.csv'): 
                file_name += '.csv'    
            file_name = get_time_now_as_str() + file_name
            
            import os
            self.file_dir = os.path.join(self.log_dir, 
                env_name, exp_name)
            
            os.makedirs(self.file_dir, exist_ok=True)
            self.csv_dir = os.path.join(self.file_dir, file_name)
            
            self.csv_file = open(self.csv_dir, 'w', encoding='utf8')
            self.csv_writer = None
        
    def record(self, key: str, val):
        self.name_to_vals[key] = val
        
    def dict_record(self, report: dict):
        self.name_to_vals.update(report)
            
    def __getitem__(self, key: str):
        return self.name_to_vals.get(key, None)
    
    def dump(self):
        print('='*30)
        for key, val in self.name_to_vals.items():
            print(f"{key:<20} : {_maybe_float_roundoff(val)}")
        
        if self.log_to_file:
            if not self.csv_writer:
                self.csv_writer=csv.DictWriter(
                        self.csv_file, fieldnames=self.name_to_vals.keys())
                self.csv_writer.writeheader()
            self.csv_writer.writerow(
                dict(map( lambda it: (it[0], _maybe_float_roundoff(it[1])),
                        self.name_to_vals.items() ))
            )
            self.csv_file.flush()
            
    def dump_file(self):
        self.csv_file.flush()
        
def _maybe_float_roundoff(n):
    if isinstance(n, float):
        return f'{n:.2f}'
    return n