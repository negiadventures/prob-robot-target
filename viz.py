import pandas as pd

df = pd.read_csv('data.csv')
import plotly.express as px

df['moves_by_examinations'] = df['moves'] / df['examinations']
df['moves_and_examinations'] = df['moves'] + df['examinations']
# pd.set_option('display.max_columns', 20)

print(pd.DataFrame(df.groupby('agent').mean().to_dict()).rename_axis('agent').reset_index())
df = df[(df.agent == 'Agent 6') | (df.agent == 'Agent 7')| (df.agent == 'Agent 8') ]
# 1.Run time
fig = px.box(df, x="agent", y="time",title="Agent 6, 7, 8 - Runtime Comparison", log_y=True, points="all", labels={
        "time": "Time (s)",
        "agent": "Agent"
    })
fig.update_xaxes(categoryorder='array', categoryarray=["Agent 6", "Agent 7", "Agent 8"])
fig.update_traces(quartilemethod="inclusive") # or "inclusive", or "linear" by default
fig.show()

# 2.move
fig = px.line(df, x="iteration", y="moves", color='agent', line_shape='spline', render_mode='svg',
              title="Agent 6, 7, 8 - Moves Comparison", labels={
        "moves": "Moves",
        "iteration": "Iteration",
        "agent": "Agent"
    })
fig.show()

# 3.examine
fig = px.line(df, x="iteration", y="examinations", color='agent', line_shape='spline', render_mode='svg',
              title="Agent 6, 7, 8 - Examinations Comparison", labels={
        "examinations": "Examinations",
        "iteration": "Iteration",
        "agent": "Agent"
    })
fig.show()
# TODO: 4.move+ examine with dim(avg)
fig = px.line(df, x="iteration", y="moves_and_examinations", color='agent', line_shape='spline', render_mode='svg',
              title="Agent 6, 7, 8  - Moves + Examinations Comparison", labels={
        "moves_and_examinations": "Moves + Examinations",
        "iteration": "Iteration",
        "agent": "Agent"
    })
fig.show()

# 6.move/examine with dim
fig = px.line(df, x="iteration", y="moves_by_examinations", color='agent', line_shape='spline', render_mode='svg',
              title="Agent 6, 7, 9 - Moves / Examinations Comparison", labels={
        "moves_by_examinations": "Moves / Examinations",
        "iteration": "Iteration",
        "agent": "Agent"
    })
fig.show()