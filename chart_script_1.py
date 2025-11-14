
import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Define box positions (x, y) for each stage
boxes = [
    {"name": "Data Collection", "subtitle": "FastF1 API", 
     "items": ["Race Results", "Qualifying Data", "Lap Telemetry", "Weather Data", "Pit Stops"],
     "x": 0.5, "y": 5.5, "color": "#DB4545"},
    
    {"name": "Data Preprocessing", "subtitle": "", 
     "items": ["Filter DNF", "Handle Missing Values", "Encode Categories"],
     "x": 0.5, "y": 4.5, "color": "#1FB8CD"},
    
    {"name": "Feature Engineering", "subtitle": "", 
     "items": ["Grid Advantage", "Relative Pace", "Team Strength", "Driver Skill", "Pit Stop Rate"],
     "x": 0.5, "y": 3.5, "color": "#9FA8B0"},
    
    {"name": "Model Training", "subtitle": "", 
     "items": ["Random Forest", "Gradient Boosting", "Cross-Validation"],
     "x": 0.5, "y": 2.5, "color": "#DB4545"},
    
    {"name": "Model Evaluation", "subtitle": "", 
     "items": ["MAE", "RMSE", "R² Score", "Feature Importance"],
     "x": 0.5, "y": 1.5, "color": "#1FB8CD"},
    
    {"name": "Race Prediction", "subtitle": "", 
     "items": ["Predict Positions", "Generate Rankings"],
     "x": 0.5, "y": 0.5, "color": "#9FA8B0"}
]

# Box dimensions
box_width = 0.7
box_height = 0.7

# Add boxes and text
for box in boxes:
    # Determine text color based on background
    text_color = "white" if box["color"] in ["#DB4545", "#1FB8CD"] else "black"
    
    # Add rectangle
    fig.add_shape(
        type="rect",
        x0=box["x"] - box_width/2,
        y0=box["y"] - box_height/2,
        x1=box["x"] + box_width/2,
        y1=box["y"] + box_height/2,
        fillcolor=box["color"],
        line=dict(color="#333333", width=3)
    )
    
    # Build text content
    text_lines = [f"<b>{box['name']}</b>"]
    if box["subtitle"]:
        text_lines.append(f"<i>{box['subtitle']}</i>")
    text_lines.append("─────────")
    text_lines.extend(box["items"])
    
    # Add text annotation
    fig.add_annotation(
        x=box["x"],
        y=box["y"],
        text="<br>".join(text_lines),
        showarrow=False,
        font=dict(size=11, color=text_color),
        align="center"
    )

# Add arrows between boxes
for i in range(len(boxes) - 1):
    fig.add_annotation(
        x=boxes[i]["x"],
        y=boxes[i]["y"] - box_height/2,
        ax=boxes[i+1]["x"],
        ay=boxes[i+1]["y"] + box_height/2,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor="#333333"
    )

# Update layout
fig.update_layout(
    title="F1 ML Prediction Workflow",
    showlegend=False,
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[0, 6]),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# Save as PNG and SVG
fig.write_image("f1_workflow.png")
fig.write_image("f1_workflow.svg", format="svg")
