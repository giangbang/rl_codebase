# import matplotlib.pyplot as plt
# import os
# import expt
# from expt.plot import GridPlot


# def plot_results(root_dir='./logging'):
    # runs = expt.get_runs(f"{root_dir}*").filter(lambda r: 'rb' not in r.name)
    
    # df = runs.to_dataframe()
    # df_names = df['name'].str.split('_', expand=True)
    # df.insert('env_name', df_names[0], True)
    # df.insert('exp_name', df_names[1], True)
    
    # g = GridPlot(y_names=df.reset_index().env_name.unique())
    
import matplotlib.pyplot as plt
import os
import pandas as pd


log_dir = 'logging'
for dir in os.listdir(log_dir):
    file = os.path.join(log_dir, dir, 'progress.csv')
    df = pd.read_csv(file)
    plt.plot(df['time.total_timesteps'], df['eval.rewards'], label=dir)
plt.legend()
plt.show()