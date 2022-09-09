
class Logger:
    def __init__(self, log_dir=None, file_name='progress.csv'):
        self.log_dir = log_dir
        self.log_to_file = log_dir is not None
        self.name_to_vals = {}
        
        if self.log_to_file:
            assert file_name.endswith('.csv')
            import os
            self.file_dir = os.path.join(self.log_dir, file_name)
            os.makedirs(log_dir, exist_ok=True)
            self.csv_file = open(self.file_dir, 'w', encoding='utf8')
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
            if isinstance(val, float):
                print(f"{key:<20} : {val:.2f}")
            else:
                print(f"{key:<20} : {val}")
        
        if self.log_to_file:
            if not self.csv_writer:
                self.csv_writer=csv.DictWriter(
                        self.csv_file, fieldnames=self.name_to_vals.keys())
                self.csv_writer.writeheader()
            self.csv_writer.writerow(self.name_to_vals)
            self.csv_file.flush()
            
    def dump_file(self):
        self.csv_file.flush()