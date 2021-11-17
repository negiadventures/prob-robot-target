import pandas as pd
import plotly.express as px

df_5 = pd.read_csv('../data_5.csv')
df_5['dim'] = 5
df_5['moves_by_examinations'] = df_5['moves'] / df_5['examinations']
df_5['moves_and_examinations'] = df_5['moves'] + df_5['examinations']
df_10 = pd.read_csv('../data_10.csv')
df_10['dim'] = 10
df_10['moves_by_examinations'] = df_10['moves'] / df_10['examinations']
df_10['moves_and_examinations'] = df_10['moves'] + df_10['examinations']
df_15 = pd.read_csv('../data_15.csv')
df_15['dim'] = 15
df_15['moves_by_examinations'] = df_15['moves'] / df_15['examinations']
df_15['moves_and_examinations'] = df_15['moves'] + df_15['examinations']
df_20 = pd.read_csv('../data_20.csv')
df_20['dim'] = 20
print(df_20)
df_20['moves_by_examinations'] = df_20['moves'] / df_20['examinations']
df_20['moves_and_examinations'] = df_20['moves'] + df_20['examinations']
df_25 = pd.read_csv('../data_25.csv')
df_25['dim'] = 25
df_25['moves_by_examinations'] = df_25['moves'] / df_25['examinations']
df_25['moves_and_examinations'] = df_25['moves'] + df_25['examinations']
df_50 = pd.read_csv('../data_50.csv')
df_50['dim'] = 50
df_50['moves_by_examinations'] = df_50['moves'] / df_50['examinations']
df_50['moves_and_examinations'] = df_50['moves'] + df_50['examinations']
df_101 = pd.read_csv('../data_101.csv')
df_101['dim'] = 101
df_101['moves_by_examinations'] = df_101['moves'] / df_101['examinations']
df_101['moves_and_examinations'] = df_101['moves'] + df_101['examinations']

df_5 = pd.DataFrame(df_5.groupby('agent').mean().to_dict())
df_5 = df_5.rename_axis('agent').reset_index()

df_10 = pd.DataFrame(df_10.groupby('agent').mean().to_dict())
df_10 = df_10.rename_axis('agent').reset_index()

df_15 = pd.DataFrame(df_15.groupby('agent').mean().to_dict())
df_15 = df_15.rename_axis('agent').reset_index()

df_20 = pd.DataFrame(df_20.groupby('agent').mean().to_dict())
df_20 = df_20.rename_axis('agent').reset_index()

df_25 = pd.DataFrame(df_25.groupby('agent').mean().to_dict())
df_25 = df_25.rename_axis('agent').reset_index()

df_50 = pd.DataFrame(df_50.groupby('agent').mean().to_dict())
df_50 = df_50.rename_axis('agent').reset_index()

df_101 = pd.DataFrame(df_101.groupby('agent').mean().to_dict())
df_101 = df_101.rename_axis('agent').reset_index()

# df_20 = df_20[df_20['dividing_factor']!='8.1']
# del df_20['iteration']
# pd.set_option('display.max_columns', 20)

frames = [df_5, df_10, df_15, df_20, df_25, df_50, df_101]

result = pd.concat(frames)
result = result[(result.agent == 6) | (result.agent == 7) | (result.agent == 8)]
# print(result.head(20))
# print(pd.DataFrame(df_20.groupby('agent').mean().to_dict()).rename_axis('dividing_factor').reset_index())
# fig = px.line(result, x="dim", y="time", color='agent', render_mode='svg', log_x=True)
# fig.show()

# 5.move+ examine with dim(avg)
fig = px.line(result, x="dim", y="moves_and_examinations", color='agent', log_x=True, render_mode='svg',
              title="Agent 6, 7, 8 - Moves + Examinations Comparison (with different dimensions)", labels={
        "moves_and_examinations": "Moves + Examinations",
        "dim": "Grid Dimension",
        "agent": "Agent"
    })
fig.show()

# 7.move/examine with dim
fig = px.line(result, x="dim", y="moves_by_examinations", color='agent', log_x=True, render_mode='svg',
              title="Agent 6, 7, 8 - Moves / Examinations Comparison (with different dimensions)", labels={
        "moves_by_examinations": "Moves / Examinations",
        "dim": "Grid Dimension",
        "agent": "Agent"
    })
fig.show()

# 8.time with dim
fig = px.line(result, x="dim", y="time", color='agent', render_mode='svg', log_x=True,
              title="Agent 6, 7, 8 - Runtime Comparison (with different dimensions)", labels={
        "time": "Time (s)",
        "dim": "Grid Dimension",
        "agent": "Agent"
    })
fig.show()
