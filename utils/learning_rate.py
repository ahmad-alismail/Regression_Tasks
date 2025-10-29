"""
learning_rate.py

A Python script containing functions to generate Plotly visualizations 
related to machine learning learning rates. These functions can be imported 
into a Jupyter Notebook or another Python script.

Functions:
- plot_loss_curves: Creates a static plot comparing different loss curve behaviors.
- plot_gradient_descent_animation: Creates an animated plot visualizing the
  impact of different learning rates on gradient descent.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_loss_curves():
    """
    Generates a static plot comparing three common loss curve behaviors:
    1. Optimal alpha: Fast convergence to a plateau.
    2. Low alpha: Very slow, linear improvement.
    3. Big alpha: Noisy and unstable training.
    
    Returns:
        go.Figure: A Plotly figure object.
    """
    np.random.seed(42)
    iters = np.arange(0, 100)

    # 1) Fast convergence (Optimal alpha)
    asymptote = 180.0
    start_val = 2400.0
    k = 0.15
    loss_fast_plateau = asymptote + (start_val - asymptote) * np.exp(-k * iters)

    # 2) Very slow improvement (Low alpha)
    start2 = 1500.0
    end2 = 500.0
    slope = (start2 - end2) / (len(iters) - 1)
    loss_slow = start2 - slope * iters

    # 3) Noisy/unstable training (Big alpha)
    base = 850.0
    noise = np.random.normal(0, 60, size=len(iters))
    spikes_idx = np.random.choice(iters, size=8, replace=False)
    noise[spikes_idx] += np.random.normal(0, 120, size=len(spikes_idx))
    loss_noisy = base + noise

    # Create combined figure
    fig = go.Figure()

    # Add all three traces
    fig.add_trace(go.Scatter(
        x=iters, 
        y=loss_fast_plateau,
        mode='lines',
        name='Optimal α: Fast Convergence',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=iters, 
        y=loss_slow,
        mode='lines',
        name='Low α: Very Slow Improvement',
        line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=iters, 
        y=loss_noisy,
        mode='lines',
        name='Big α: Noisy / Unstable',
        line=dict(color='green', width=2)
    ))

    # Update layout
    fig.update_layout(
        title="Comparison of Different Loss Curve Behaviors",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=800,
        height=500,
        template="plotly_white"
    )

    return fig


def plot_gradient_descent_animation():
    """
    Generates an animated plot showing the path of gradient descent on a
    simple convex function for small, optimal, and large learning rates.
    
    The animation consists of two subplots:
    1. The descent path on the cost function curve.
    2. The cost value over iterations.
    
    Returns:
        go.Figure: A Plotly figure object with animation frames and controls.
    """
    # --- 1. Define the Cost Function and its Gradient ---
    # A simple convex function: J(β) = β^2
    def cost_function(beta):
        return beta**2

    # The derivative (gradient) of J(β) = β^2 is 2β
    def gradient(beta):
        return 2 * beta

    # --- 2. Simulate Gradient Descent for Different Learning Rates ---
    initial_beta = 4.0
    n_iterations = 40

    # Define the learning rates and their descriptive labels for the legend
    scenarios = {
        "Small": {"alpha": 0.03, "color": "royalblue", "label": "Small learning rate: Slow Improvement"},
        "Optimal": {"alpha": 0.2, "color": "green", "label": "Good learning rate: Fast Convergence"},
        "Large": {"alpha": 0.985, "color": "firebrick", "label": "Big learning rate: Unstable"},
    }

    # Store the history of beta values and costs for each scenario
    history = {}
    for name, props in scenarios.items():
        betas = [initial_beta]
        costs = [cost_function(initial_beta)]
        current_beta = initial_beta
        for _ in range(n_iterations):
            grad = gradient(current_beta)
            current_beta = current_beta - props["alpha"] * grad
            betas.append(current_beta)
            costs.append(cost_function(current_beta))
        history[name] = {'betas': np.array(betas), 'costs': np.array(costs)}

    # --- 3. Create the Animation ---
    fig = make_subplots(
        rows=1, cols=2,
        #subplot_titles=("Gradient Descent Path on Cost Function", "Cost vs. Iterations")
    )

    # Define the static U-shaped curve (with no legend entry)
    fig.add_trace(go.Scatter(
        x=np.linspace(-4.5, 4.5, 100), y=cost_function(np.linspace(-4.5, 4.5, 100)),
        mode='lines', line=dict(color='black', width=2),
        showlegend=False  # Hide this from the legend
    ), row=1, col=1)

    # Initialize the plots with the starting position (Frame 0)
    for name, props in scenarios.items():
        # Left plot: path lines (hidden from legend)
        fig.add_trace(go.Scatter(
            x=[initial_beta], y=[cost_function(initial_beta)],
            mode='lines', line=dict(color=props["color"], width=2, dash='dot'),
            showlegend=False
        ), row=1, col=1)
        
        # Left plot: current position markers (hidden from legend)
        fig.add_trace(go.Scatter(
            x=[initial_beta], y=[cost_function(initial_beta)],
            mode='markers', marker=dict(color=props["color"], size=12),
            showlegend=False
        ), row=1, col=1)

        # Right plot: cost vs. iterations lines (THIS IS THE ONLY PART WITH A LEGEND)
        fig.add_trace(go.Scatter(
            x=[0], y=[cost_function(initial_beta)],
            mode='lines+markers', line=dict(color=props["color"], width=2.5),
            name=props["label"]  # Use the descriptive label here
        ), row=1, col=2)

    # --- Create the animation frames ---
    frames = []
    for i in range(1, n_iterations + 1):
        frame_data = []
        # Loop through scenarios to update traces in the correct order
        for name in scenarios.keys():
            frame_data.append(go.Scatter(x=history[name]['betas'][:i+1], y=history[name]['costs'][:i+1])) # Path
            frame_data.append(go.Scatter(x=[history[name]['betas'][i]], y=[history[name]['costs'][i]]))   # Marker
            frame_data.append(go.Scatter(x=list(range(i + 1)), y=history[name]['costs'][:i+1]))        # Cost line
        
        frames.append(go.Frame(
            data=frame_data,
            name=f"iter_{i}",
            # These indices correspond to the traces we added in the loop above
            traces=[1, 2, 3, 4, 5, 6, 7, 8, 9] 
        ))

    fig.frames = frames

    # --- Configure layout and animation settings ---
    fig.update_layout(
        title_text="<b>Visualizing the Impact of Learning Rate in Gradient Descent</b>",
        xaxis=dict(range=[-4.5, 4.5], title="Parameter (β)"),
        yaxis=dict(range=[-5, 20], title="Cost J(β)"),
        xaxis2=dict(title="Iteration"),
        yaxis2=dict(title="Cost"),
        template="plotly_white",
        height=500,
        
        # Configure the legend to be clean and centered at the top
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        
        updatemenus=[{
            "type": "buttons", "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 400, "redraw": False}, "transition": {"duration": 250, "easing": "quadratic-in-out"}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]}
            ],
            "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False,
            "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
        }]
    )

    return fig


# This block allows the script to be run directly to display the plots
if __name__ == '__main__':
    print("Running learning_rate.py directly. Displaying plots...")
    
    print("Displaying static loss curve comparison plot...")
    fig1 = plot_loss_curves()
    fig1.show()
    
    print("Displaying animated gradient descent plot...")
    fig2 = plot_gradient_descent_animation()
    fig2.show()