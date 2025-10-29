import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def generate_loss_curves(n_epochs=100, n_batches_per_epoch=50):
    """
    Generate synthetic loss curves for SGD and Mini-batch GD
    """
    np.random.seed(42)
    
    # Parameters for loss curve generation
    initial_loss = 2.5
    final_loss = 0.1
    
    # Generate epochs
    epochs = np.arange(n_epochs)
    
    # Mini-batch GD: Smooth exponential decay with small noise
    minibatch_loss = initial_loss * np.exp(-0.05 * epochs) + final_loss
    minibatch_noise = np.random.normal(0, 0.02, n_epochs)
    minibatch_loss += minibatch_noise
    minibatch_loss = np.maximum(minibatch_loss, final_loss)  # Ensure it doesn't go below final loss
    
    # SGD: Much more noisy, with higher variance
    sgd_base = initial_loss * np.exp(-0.04 * epochs) + final_loss  # Slightly slower convergence
    sgd_noise = np.random.normal(0, 0.15, n_epochs)  # Much higher noise
    
    # Add some additional oscillations for SGD
    oscillation = 0.1 * np.sin(0.3 * epochs) * np.exp(-0.02 * epochs)
    sgd_loss = sgd_base + sgd_noise + oscillation
    sgd_loss = np.maximum(sgd_loss, final_loss * 0.8)  # Allow it to go slightly lower sometimes
    
    return epochs, sgd_loss, minibatch_loss

def create_animated_plot():
    """
    Create an animated plot comparing SGD vs Mini-batch GD loss curves
    """
    epochs, sgd_loss, minibatch_loss = generate_loss_curves()
    
    # Create figure with initial empty traces
    fig = go.Figure()
    
    # Add initial traces (empty)
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        name='SGD (Stochastic)',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))
    
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        mode='lines+markers',
        name='Mini-batch GD',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Create frames for animation
    frames = []
    for i in range(1, len(epochs) + 1):
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=epochs[:i],
                    y=sgd_loss[:i],
                    mode='lines+markers',
                    name='SGD (Stochastic)',
                    line=dict(color='red', width=2),
                    marker=dict(size=3)
                ),
                go.Scatter(
                    x=epochs[:i],
                    y=minibatch_loss[:i],
                    mode='lines+markers',
                    name='Mini-batch GD',
                    line=dict(color='blue', width=2),
                    marker=dict(size=3)
                )
            ],
            name=str(i)
        ))
    
    fig.frames = frames
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'SGD vs Mini-batch Gradient Descent: Loss Curves Comparison',
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title='Epoch',
        yaxis_title='Loss',
        xaxis=dict(range=[0, len(epochs)]),
        yaxis=dict(range=[0, max(max(sgd_loss), max(minibatch_loss)) * 1.1]),
        width=900,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="<b>Key Differences:</b><br>" +
                     "🔴 <b>SGD</b>: High variance, noisy updates<br>" +
                     "🔵 <b>Mini-batch</b>: Smooth, stable convergence",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"frame": {"duration": 150, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 50}}],
                        label="▶️ Play",
                        method="animate"
                    ),
                    dict(
                        args=[{"frame": {"duration": 0, "redraw": False},
                              "mode": "immediate",
                              "transition": {"duration": 0}}],
                        label="⏸️ Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            ),
        ],
        # Add slider for manual control
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Epoch: "},
            pad={"t": 50},
            steps=[dict(
                args=[
                    [str(k)],
                    {"frame": {"duration": 0, "redraw": True},
                     "mode": "immediate",
                     "transition": {"duration": 0}}
                ],
                label=str(k),
                method="animate"
            ) for k in range(1, len(epochs) + 1)]
        )]
    )
    
    return fig



def main():
    """
    Main function to create and display animated plot
    """
    
    
    # Create animated plot
    animated_fig = create_animated_plot()
    animated_fig.show()
    
    