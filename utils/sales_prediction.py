# sales_prediction.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Docstrings are used to explain what a function does. It's a great habit!
def train_sales_model(df: pd.DataFrame):
    """
    Trains a multiple linear regression model on TV and Radio advertising data.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'TV', 'radio', and 'sales' columns.

    Returns:
        tuple: A tuple containing:
            - reg (LinearRegression): The fitted scikit-learn model object.
            - X (np.ndarray): The feature matrix used for training (TV and radio).
            - sales (np.ndarray): The target variable array (sales).
    """
    print("--- Training Model ---")
    # Extract the variables for our model
    TV = df['TV'].values
    radio = df['radio'].values
    sales = df['sales'].values

    # Fit the multiple linear regression model
    X = np.column_stack([TV, radio])  # Feature matrix
    reg = LinearRegression().fit(X, sales)

    # Get the coefficients and print a summary
    beta_0 = reg.intercept_
    beta_1, beta_2 = reg.coef_

    print(f"Fitted Model: Sales = {beta_0:.3f} + {beta_1:.3f}·TV + {beta_2:.3f}·Radio")
    print(f"R² Score: {reg.score(X, sales):.3f}\n")
    
    return reg, X, sales

def plot_regression_plane(reg_model, X, sales):
    """
    Generates a 3D plot showing actual sales data and the fitted regression plane.

    Args:
        reg_model (LinearRegression): The fitted scikit-learn model object.
        X (np.ndarray): The feature matrix (TV and radio).
        sales (np.ndarray): The target variable array (sales).

    Returns:
        go.Figure: A Plotly figure object that can be displayed with .show()
    """
    print("--- Generating Regression Plane Plot ---")
    # Extract individual features for clarity
    TV = X[:, 0]
    radio = X[:, 1]
    
    # Get model coefficients
    beta_0 = reg_model.intercept_
    beta_1, beta_2 = reg_model.coef_

    # Create meshgrid for the regression plane
    tv_range = np.linspace(TV.min(), TV.max(), 20)
    radio_range = np.linspace(radio.min(), radio.max(), 20)
    tv_mesh, radio_mesh = np.meshgrid(tv_range, radio_range)
    sales_mesh = beta_0 + beta_1 * tv_mesh + beta_2 * radio_mesh

    # Create the 3D scatter plot
    fig = go.Figure()

    # Add scatter points for actual data
    fig.add_trace(go.Scatter3d(
        x=TV, y=radio, z=sales, mode='markers',
        marker=dict(size=5, color=sales, colorscale='Viridis', colorbar=dict(title="Sales")),
        name='Actual Data',
        text=[f'TV: {tv:.1f}<br>Radio: {r:.1f}<br>Sales: {s:.1f}' for tv, r, s in zip(TV, radio, sales)],
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))

    # Add the regression plane
    fig.add_trace(go.Surface(
        x=tv_mesh, y=radio_mesh, z=sales_mesh,
        opacity=0.7, colorscale='RdYlBu', showscale=False, name='Regression Plane'
    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text='3D Multiple Linear Regression: Sales vs TV and Radio<br>' +
                 f'<sub>Model: Sales = {beta_0:.2f} + {beta_1:.3f}·TV + {beta_2:.3f}·Radio (R² = {reg_model.score(X, sales):.3f})</sub>',
            x=0.5
        ),
        scene=dict(
            xaxis_title='TV Advertising ($)',
            yaxis_title='Radio Advertising ($)', 
            zaxis_title='Sales ($1000s)'
        ),
        width=900, height=700
    )
    
    return fig

def plot_residuals_3d(reg_model, X, sales):
    """
    Generates a 3D plot of the model's residuals.

    Args:
        reg_model (LinearRegression): The fitted scikit-learn model object.
        X (np.ndarray): The feature matrix (TV and radio).
        sales (np.ndarray): The target variable array (sales).

    Returns:
        go.Figure: A Plotly figure object that can be displayed with .show()
    """
    print("--- Generating 3D Residuals Plot ---")
    # Extract individual features for clarity
    TV = X[:, 0]
    radio = X[:, 1]

    # Calculate residuals
    predictions = reg_model.predict(X)
    residuals = sales - predictions

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=TV, y=radio, z=residuals, mode='markers',
        marker=dict(size=4, color=residuals, colorscale='RdBu', colorbar=dict(title="Residuals")),
        name='Residuals',
        text=[f'TV: {tv:.1f}<br>Radio: {r:.1f}<br>Residual: {res:.2f}' for tv, r, res in zip(TV, radio, residuals)],
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    # Add zero plane for reference
    tv_mesh, radio_mesh = np.meshgrid(np.linspace(TV.min(), TV.max(), 2), np.linspace(radio.min(), radio.max(), 2))
    zero_plane = np.zeros_like(tv_mesh)
    
    fig.add_trace(go.Surface(
        x=tv_mesh, y=radio_mesh, z=zero_plane,
        opacity=0.3, showscale=False, colorscale=[[0, 'gray'], [1, 'gray']], name='Zero Plane'
    ))

    fig.update_layout(
        title='3D Residuals Plot: Sales vs TV and Radio',
        scene=dict(
            xaxis_title='TV Advertising ($)',
            yaxis_title='Radio Advertising ($)',
            zaxis_title='Residuals ($)'
        ),
        width=900, height=700
    )
    
    return fig