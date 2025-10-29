"""
Complete Regression Visualization Script
========================================

This script contains functions to create beautiful visualizations for different types of regression:
1. Simple Linear Regression
2. Multiple Linear Regression (3D)
3. Polynomial Regression
4. Multiple Polynomial Regression (3D)

Requirements:
- numpy
- plotly
- scikit-learn
- pandas (optional, for data handling)

Usage:
Run individual functions or use the main() function to display all plots.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


def simple_linear_regression(n_samples=50, noise_level=1.0, random_seed=42, show_plot=True):
    """
    Simple Linear Regression: y = mx + b
    
    Parameters:
    -----------
    n_samples : int, default=50
        Number of data points
    noise_level : float, default=1.0
        Amount of noise to add
    random_seed : int, default=42
        Random seed for reproducibility
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : plotly figure
    model : sklearn model
    """
    np.random.seed(random_seed)
    
    # Generate sample data
    X = np.random.uniform(0, 10, n_samples)
    noise = np.random.normal(0, noise_level, n_samples)
    y = 2 * X + 3 + noise  # True relationship: y = 2x + 3
    
    # Fit linear regression
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_reshaped, y)
    
    # Generate points for the line
    X_line = np.linspace(0, 10, 100)
    y_pred_line = model.predict(X_line.reshape(-1, 1))
    
    # Create the plot
    fig = go.Figure()
    
    # Add scatter plot for data points
    fig.add_trace(go.Scatter(
        x=X, 
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(
            color='rgba(54, 162, 235, 0.8)',
            size=8,
            line=dict(width=1, color='white')
        ),
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    # Add regression line
    fig.add_trace(go.Scatter(
        x=X_line,
        y=y_pred_line,
        mode='lines',
        name=f'Best Fit Line',
        line=dict(color='red', width=3)
    ))
    
    # Update layout
    r2 = model.score(X_reshaped, y)
    fig.update_layout(
        title=dict(
            text=f'Simple Linear Regression: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}',
            x=0.5,
            font=dict(size=18)
        ),
        xaxis_title='X (Input Variable)',
        yaxis_title='Y (Output Variable)',
        template='plotly_white',
        width=800,
        height=500,
        showlegend=True
    )
    
    # Add R² annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f'R² = {r2:.3f}<br>Slope = {model.coef_[0]:.3f}<br>Intercept = {model.intercept_:.3f}',
        showarrow=False,
        align='left',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    if show_plot:
        fig.show()
        print(f"Simple Linear Regression - R² Score: {r2:.3f}")
    
    return fig, model


def multiple_linear_regression(n_samples=100, noise_level=1.0, random_seed=42, show_plot=True):
    """
    Multiple Linear Regression: z = b0 + b1*x1 + b2*x2
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of data points
    noise_level : float, default=1.0
        Amount of noise to add
    random_seed : int, default=42
        Random seed for reproducibility
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : plotly figure
    model : sklearn model
    """
    np.random.seed(random_seed)
    
    # Generate sample data with 2 input variables
    X1 = np.random.uniform(0, 10, n_samples)
    X2 = np.random.uniform(0, 10, n_samples)
    noise = np.random.normal(0, noise_level, n_samples)
    
    # True relationship: z = 2*x1 + 3*x2 + 5
    z = 2 * X1 + 3 * X2 + 5 + noise
    
    # Prepare data for sklearn
    X = np.column_stack((X1, X2))
    
    # Fit multiple linear regression
    model = LinearRegression()
    model.fit(X, z)
    
    # Create meshgrid for the plane
    x1_range = np.linspace(X1.min(), X1.max(), 20)
    x2_range = np.linspace(X2.min(), X2.max(), 20)
    X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
    
    # Calculate predicted values for the mesh
    mesh_points = np.column_stack((X1_mesh.ravel(), X2_mesh.ravel()))
    Z_pred = model.predict(mesh_points).reshape(X1_mesh.shape)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter3d(
        x=X1, y=X2, z=z,
        mode='markers',
        name='Data Points',
        marker=dict(
            color='blue',
            size=5,
            opacity=0.8
        ),
        hovertemplate='X₁: %{x:.2f}<br>X₂: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))
    
    # Add regression plane
    fig.add_trace(go.Surface(
        x=X1_mesh,
        y=X2_mesh,
        z=Z_pred,
        name='Regression Plane',
        opacity=0.7,
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="Z Value", x=1.02)
    ))
    
    # Update layout
    r2 = model.score(X, z)
    fig.update_layout(
        title=dict(
            text=f'Multiple Linear Regression<br>z = {model.intercept_:.2f} + {model.coef_[0]:.2f}x₁ + {model.coef_[1]:.2f}x₂',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X₁ (Variable 1)',
            yaxis_title='X₂ (Variable 2)',
            zaxis_title='Z (Output)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=900,
        height=600
    )
    
    # Add R² annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f'R² = {r2:.3f}',
        showarrow=False,
        font=dict(size=14),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    if show_plot:
        fig.show()
        print(f"Multiple Linear Regression - R² Score: {r2:.3f}")
    
    return fig, model



def polynomial_regression(degree=2, n_samples=25, noise_level=0.3, x_range=(0, 10), 
                             random_seed=42, width=800, height=500):
    """
    Create a polynomial regression plot with Plotly
    
    Parameters:
    -----------
    degree : int, default=2
        Degree of the polynomial (1, 2, 3, etc.)
    n_samples : int, default=25  
        Number of data points to generate
    noise_level : float, default=0.3
        Amount of noise to add to the data
    x_range : tuple, default=(0, 10)
        Range of x values (min, max)
    random_seed : int, default=42
        Random seed for reproducible results
    width : int, default=800
        Width of the plot in pixels
    height : int, default=500
        Height of the plot in pixels
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The polynomial regression plot
    model : sklearn.pipeline.Pipeline
        The fitted polynomial regression model
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate sample data
    X = np.random.uniform(x_range[0], x_range[1], n_samples)
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Create true polynomial relationship based on degree
    if degree == 1:
        y = 2 * X + 3 + noise
        true_equation = "y = 2x + 3"
    elif degree == 2:
        y = 0.1 * X**2 - X + 5 + noise
        true_equation = "y = 0.1x² - x + 5"
    elif degree == 3:
        y = 0.02 * X**3 - 0.2 * X**2 + X + 2 + noise
        true_equation = "y = 0.02x³ - 0.2x² + x + 2"
    else:
        # General case for higher degrees
        y = 0.1 * X**2 - 0.5 * X + 3 + noise
        true_equation = f"Polynomial degree {degree}"
    
    # Fit polynomial regression
    X_reshaped = X.reshape(-1, 1)
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X_reshaped, y)
    
    # Generate smooth curve for the fitted line
    X_smooth = np.linspace(x_range[0], x_range[1], 200)
    y_smooth = poly_model.predict(X_smooth.reshape(-1, 1))
    
    # Create the plot
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter(
        x=X, 
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(
            color='#1f77b4',  # Blue color
            size=8,
            opacity=0.7
        ),
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    # Add polynomial fit curve
    fig.add_trace(go.Scatter(
        x=X_smooth,
        y=y_smooth,
        mode='lines',
        name='Polynomial Fit',
        line=dict(
            color='red',
            width=3
        )
    ))
    
    # Create title with fitted equation
    coefficients = poly_model.named_steps['linearregression'].coef_
    intercept = poly_model.named_steps['linearregression'].intercept_
    r2_score = poly_model.score(X_reshaped, y)
    
    # Build equation string
    if degree == 1:
        equation = f"y = {coefficients[1]:.2f}x + {intercept:.2f}"
    elif degree == 2:
        equation = f"y = {coefficients[2]:.2f}x² + {coefficients[1]:.2f}x + {intercept:.2f}"
    elif degree == 3:
        equation = f"y = {coefficients[3]:.3f}x³ + {coefficients[2]:.2f}x² + {coefficients[1]:.2f}x + {intercept:.2f}"
    else:
        equation = f"Polynomial Regression (degree {degree})"
    
    # Update layout
    fig.update_layout(
        title=f'Polynomial Regression: {equation}',
        xaxis_title='X (Input Variable)',
        yaxis_title='Y (Output Variable)',
        template='plotly_white',
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    # Show grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Add R² score as annotation
    fig.add_annotation(
        x=0.98, y=0.02,
        xref='paper', yref='paper',
        text=f'R² = {r2_score:.3f}',
        showarrow=False,
        align='right',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    # Display the plot
    fig.show()
    
    # Print model information
    print(f"Model Information:")
    print(f"Polynomial degree: {degree}")
    print(f"Number of samples: {n_samples}")
    print(f"R² Score: {r2_score:.3f}")
    print(f"Fitted equation: {equation}")
    
    return fig, poly_model


def multiple_polynomial_regression(n_samples=100, noise_level=0.3, random_seed=42, show_plot=True):
    """
    Multiple Polynomial Regression with interaction terms
    
    Parameters:
    -----------
    n_samples : int, default=100
        Number of data points
    noise_level : float, default=0.3
        Amount of noise to add
    random_seed : int, default=42
        Random seed for reproducibility
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : plotly figure
    model : sklearn pipeline
    """
    np.random.seed(random_seed)
    
    # Generate sample data
    X1 = np.random.uniform(-2, 2, n_samples)
    X2 = np.random.uniform(-2, 2, n_samples)
    noise = np.random.normal(0, noise_level, n_samples)
    
    # True relationship: z = x1² + x2² + x1*x2 + 2
    z = X1**2 + X2**2 + X1*X2 + 2 + noise
    
    # Prepare data
    X = np.column_stack((X1, X2))
    
    # Create polynomial features (degree 2 with interaction terms)
    poly_model = make_pipeline(PolynomialFeatures(2, include_bias=True), LinearRegression())
    poly_model.fit(X, z)
    
    # Create meshgrid for surface
    x1_range = np.linspace(-2, 2, 30)
    x2_range = np.linspace(-2, 2, 30)
    X1_mesh, X2_mesh = np.meshgrid(x1_range, x2_range)
    mesh_points = np.column_stack((X1_mesh.ravel(), X2_mesh.ravel()))
    
    # Predict with polynomial model
    Z_poly = poly_model.predict(mesh_points).reshape(X1_mesh.shape)
    
    # Create 3D plot
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter3d(
        x=X1, y=X2, z=z,
        mode='markers',
        name='Data Points',
        marker=dict(
            color='blue', 
            size=5, 
            opacity=0.8
        ),
        hovertemplate='X₁: %{x:.2f}<br>X₂: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
    ))
    
    # Add polynomial surface
    fig.add_trace(go.Surface(
        x=X1_mesh, 
        y=X2_mesh, 
        z=Z_poly,
        name='Polynomial Surface',
        opacity=0.7,
        colorscale='Reds',
        showscale=True,
        colorbar=dict(
            title="Z Value",
            x=1.02
        )
    ))
    
    # Calculate R² score
    poly_r2 = poly_model.score(X, z)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Multiple Polynomial Regression<br>z = x₁² + x₂² + x₁x₂ + constant',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='X₁ (Variable 1)',
            yaxis_title='X₂ (Variable 2)', 
            zaxis_title='Z (Output)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        width=900,
        height=700
    )
    
    # Add R² annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f'R² = {poly_r2:.3f}',
        showarrow=False,
        font=dict(size=14),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1
    )
    
    if show_plot:
        fig.show()
        print(f"Multiple Polynomial Regression - R² Score: {poly_r2:.3f}")
        
        # Print feature information
        poly_features = poly_model.named_steps['polynomialfeatures']
        feature_names = poly_features.get_feature_names_out(['X1', 'X2'])
        coefficients = poly_model.named_steps['linearregression'].coef_
        
        print("\nPolynomial Features and Coefficients:")
        for name, coef in zip(feature_names, coefficients):
            print(f"  {name}: {coef:.3f}")
    
    return fig, poly_model


def comparison_dashboard():
    """
    Create a comparison dashboard showing all regression types
    """
    print("=" * 60)
    print("REGRESSION COMPARISON DASHBOARD")
    print("=" * 60)
    
    # Generate all plots without showing them individually
    print("\n1. Generating Simple Linear Regression...")
    fig1, model1 = simple_linear_regression(show_plot=False)
    
    print("2. Generating Multiple Linear Regression...")
    fig2, model2 = multiple_linear_regression(show_plot=False)
    
    print("3. Generating Polynomial Regression...")
    fig3, model3 = polynomial_regression(degree=2, show_plot=False)
    
    print("4. Generating Multiple Polynomial Regression...")
    fig4, model4 = multiple_polynomial_regression(show_plot=False)
    
    # Show all plots
    print("\nDisplaying all regression plots...")
    fig1.show()
    fig2.show() 
    fig3.show()
    fig4.show()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("All four regression types have been displayed:")
    print("✓ Simple Linear Regression")
    print("✓ Multiple Linear Regression (3D)")
    print("✓ Polynomial Regression")
    print("✓ Multiple Polynomial Regression (3D)")
    
    return fig1, fig2, fig3, fig4


def main():
    """
    Main function to demonstrate all regression types
    """
    print("Regression Visualization Script")
    print("==============================")
    print()
    
    while True:
        print("Choose an option:")
        print("1. Simple Linear Regression")
        print("2. Multiple Linear Regression")
        print("3. Polynomial Regression")
        print("4. Multiple Polynomial Regression")
        print("5. Show All Plots (Comparison Dashboard)")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            print("\nCreating Simple Linear Regression...")
            simple_linear_regression()
            
        elif choice == '2':
            print("\nCreating Multiple Linear Regression...")
            multiple_linear_regression()
            
        elif choice == '3':
            degree = input("Enter polynomial degree (default=2): ").strip()
            degree = int(degree) if degree.isdigit() else 2
            print(f"\nCreating Polynomial Regression (degree {degree})...")
            polynomial_regression(degree=degree)
            
        elif choice == '4':
            print("\nCreating Multiple Polynomial Regression...")
            multiple_polynomial_regression()
            
        elif choice == '5':
            print("\nCreating Comparison Dashboard...")
            comparison_dashboard()
            
        elif choice == '6':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")
        
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    # Example usage - you can comment out the main() call and use individual functions
    
    # Run the interactive menu
    main()
    
    # Or call individual functions directly:
    # simple_linear_regression()
    # multiple_linear_regression()
    # polynomial_regression(degree=3)
    # multiple_polynomial_regression()
    # comparison_dashboard()