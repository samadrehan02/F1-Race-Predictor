
import plotly.graph_objects as go
import pandas as pd

# Data provided
data = {
    "features": ["GridPosition", "TeamEncoded", "QualiPosition", "DriverEncoded", 
                 "AverageLapTime", "FastestLapTime", "GridAdvantage", "TeamStrength"],
    "importance": [28, 18, 15, 12, 10, 8, 5, 4]
}

# Create DataFrame and sort by importance (highest to lowest)
df = pd.DataFrame(data)
df = df.sort_values('importance', ascending=True)  # Ascending for horizontal bars (bottom to top)

# Abbreviate feature names to meet 15 character limit
feature_labels = {
    "GridPosition": "Grid Pos",
    "TeamEncoded": "Team Encoded",
    "QualiPosition": "Quali Pos",
    "DriverEncoded": "Driver Encoded",
    "AverageLapTime": "Avg Lap Time",
    "FastestLapTime": "Fast Lap Time",
    "GridAdvantage": "Grid Advant",
    "TeamStrength": "Team Strength"
}
df['feature_labels'] = df['features'].map(feature_labels)

# Use brand colors
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325', '#944454']

# Create horizontal bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df['importance'],
    y=df['feature_labels'],
    orientation='h',
    marker=dict(color=colors[:len(df)]),
    text=[f'{val}%' for val in df['importance']],
    textposition='outside',
    hovertemplate='%{y}<br>%{x}%<extra></extra>'
))

# Update layout
fig.update_layout(
    title="F1 Model Feature Importance",
    xaxis_title="Importance (%)",
    yaxis_title="Feature",
    showlegend=False
)

# Update x-axis to show percentages
fig.update_xaxes(range=[0, 35], ticksuffix='%')

# Apply cliponaxis=False for bars
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('feature_importance.png')
fig.write_image('feature_importance.svg', format='svg')
