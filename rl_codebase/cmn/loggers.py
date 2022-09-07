
class Logger:
    def _init__(self, log_dir=None, file_name='progress.csv'):
        self.log_dir = log_dir
        self.log_to_file = log_dir is not None
        self.name_to_vals = {}
        
        if self.log_to_file:
            assert file_name.endswith('.csv')
            import os
            self.file_dir = os.path.join(self.log_dir, file_name)
            os.makedirs(log_dir, exist_ok=True)
        
    def record(self, key: str, val):
        if key not in self.name_to_vals:
            self.name_to_vals[key] = []
        
        self.name_to_vals[key].append(val)
        
    def dict_record(self, report: dict):
        for key, val in report.items():
            self.record(key, val)
            
    def __getitem__(self, key: str):
        return self.name_to_vals.get(key, [])
        
    def to_df(self):
        import pandas as pd
        return pd.DataFrame(self.name_to_vals)
    
    def dump(self):
        for key, val in self.name_to_vals.items():
            print(f"{key} : {val[-1]:.2f}")
            
    def dump_file(self):
        df = self.to_df()
        df.to_csv(self.file_dir)