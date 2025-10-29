# File: gradient_descent.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_regression_animation(X_data, y_data, fixed_intercept=8, slope_range=(-1.5, 5.5, 80)):
    """
    Creates a side-by-side animated Plotly visualization of a regression line
    fit and its corresponding cost function.

    Args:
        X_data (np.array): The independent variable data.
        y_data (np.array): The dependent variable data.
        fixed_intercept (float, optional): The fixed intercept (β₀) to use for all lines.
                                           Defaults to 15.
        slope_range (tuple, optional): A tuple (start, end, num_steps) defining the range
                                       of slopes (β₁) for the animation.
                                       Defaults to (-1.5, 5.5, 80).

    Returns:
        go.Figure: A Plotly figure object containing the complete animation.
    """

    # --- Internal Helper Functions ---
    def _predict(beta_1, X, beta_0):
        return beta_1 * X + beta_0

    def _mse_cost(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def _cost_gradient(beta_1, X, y_true, beta_0):
        y_pred = _predict(beta_1, X, beta_0)
        return (-2 / X.shape[0]) * np.sum(X * (y_true - y_pred))

    # --- Pre-computation for Plots ---
    beta_1_values_for_cost_curve = np.linspace(slope_range[0], slope_range[1], 200)
    costs = [_mse_cost(y_data, _predict(b1, X_data, fixed_intercept)) for b1 in beta_1_values_for_cost_curve]
    
    optimal_beta_1_index = np.argmin(costs)
    optimal_beta_1 = beta_1_values_for_cost_curve[optimal_beta_1_index]
    min_cost = costs[optimal_beta_1_index]

    # --- Create the Figure with Subplots ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Fitting the Regression Line", "Cost Function (MSE)"),
        specs=[[{"type": "xy"}, {"type": "xy"}]]
    )

    # --- Left Plot: Regression Fit ---
    fig.add_trace(go.Scatter(x=X_data, y=y_data, mode='markers', name='Data Points', marker=dict(color='rgba(100, 100, 180, .7)', size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_data, y=_predict(optimal_beta_1, X_data, fixed_intercept), mode='lines', name='Optimal Fit (Fixed Intercept)', line=dict(color='green', dash='dash', width=3)), row=1, col=1)
    
    # Placeholder for the animated line
    initial_beta_1 = slope_range[0]
    fig.add_trace(go.Scatter(x=X_data, y=_predict(initial_beta_1, X_data, fixed_intercept), mode='lines', name='Current Fit', line=dict(color='red', width=3)), row=1, col=1)

    # --- Right Plot: Cost Function ---
    fig.add_trace(go.Scatter(x=beta_1_values_for_cost_curve, y=costs, mode='lines', name='Cost Function', line=dict(color='rgba(150, 150, 150, .8)')), row=1, col=2)
    fig.add_trace(go.Scatter(x=[optimal_beta_1], y=[min_cost], mode='markers', name='Minimum Cost', marker=dict(color='green', size=12, symbol='star')), row=1, col=2)
    
    # Placeholders for animated point and tangent
    initial_cost = _mse_cost(y_data, _predict(initial_beta_1, X_data, fixed_intercept))
    fig.add_trace(go.Scatter(x=[initial_beta_1], y=[initial_cost], mode='markers', name='Current Cost', marker=dict(color='red', size=12)), row=1, col=2)
    
    initial_grad = _cost_gradient(initial_beta_1, X_data, y_data, fixed_intercept)
    tangent_x = np.array([initial_beta_1 - 0.5, initial_beta_1 + 0.5])
    tangent_y = initial_grad * (tangent_x - initial_beta_1) + initial_cost
    fig.add_trace(go.Scatter(x=tangent_x, y=tangent_y, mode='lines', name='Gradient (Descent Direction)', line=dict(color='orange', width=2, dash='dot')), row=1, col=2)

    # --- Create Animation Frames ---
    animation_beta_1s = np.linspace(slope_range[0], slope_range[1], slope_range[2])
    frames = []
    for beta_1_step in animation_beta_1s:
        y_pred_step = _predict(beta_1_step, X_data, fixed_intercept)
        cost_step = _mse_cost(y_data, y_pred_step)
        grad_step = _cost_gradient(beta_1_step, X_data, y_data, fixed_intercept)
        tangent_x_step = np.array([beta_1_step - 0.5, beta_1_step + 0.5])
        tangent_y_step = grad_step * (tangent_x_step - beta_1_step) + cost_step

        frames.append(go.Frame(
            name=f"β₁={beta_1_step:.2f}",
            data=[go.Scatter(y=y_pred_step), go.Scatter(x=[beta_1_step], y=[cost_step]), go.Scatter(x=tangent_x_step, y=tangent_y_step)],
            traces=[2, 5, 6]
        ))
    fig.frames = frames

    # --- Final Layout and Animation Controls ---
    play_button = dict(type="buttons", showactive=False, buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 100, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}]), dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])])
    slider = dict(steps=[dict(method='animate', args=[[f.name], {"frame": {"duration": 100, "redraw": False}, "mode": "immediate"}], label=f.name) for f in frames], transition={"duration": 0}, x=0.1, len=0.9)

    fig.update_layout(
        title_text="Visualizing Linear Regression and Gradient Descent", title_x=0.5,
        xaxis=dict(title="X values"), yaxis=dict(title="Y values"),
        xaxis2=dict(title="Slope (β₁)"), yaxis2=dict(title="Cost (Mean Squared Error)"),
        updatemenus=[play_button], sliders=[slider],
        height=600, width=1200, showlegend=True, legend=dict(x=1.02, y=1, xanchor="left", yanchor="top")
    )

    return fig

# This block allows the script to be run directly for testing purposes.
if __name__ == '__main__':
    # Generate some sample data
    print("Running script in standalone mode for testing...")
    TRUE_BETA_1 = 2
    TRUE_BETA_0 = 5
    X_sample = np.linspace(0, 10, 50)
    np.random.seed(42)
    noise_sample = np.random.normal(0, 5, X_sample.shape[0])
    y_sample = TRUE_BETA_1 * X_sample + TRUE_BETA_0 + noise_sample

    # Create the figure by calling the function
    animation_fig = create_regression_animation(X_sample, y_sample)

    # Show the figure
    animation_fig.show()