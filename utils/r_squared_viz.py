import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from ipywidgets import interact, FloatSlider, Button, VBox, HBox, Output, HTML
import IPython.display as display

class RSquaredVisualizer:
    def __init__(self, n_points=25, x_range=(0, 20), y_range=(2, 10), noise_level=1.5):
        """
        Initialize the R² visualizer.
        
        Parameters:
        -----------
        n_points : int
            Number of data points to generate
        x_range : tuple
            Range for x values (min, max)
        y_range : tuple
            Range for y values (min, max)
        noise_level : float
            Amount of noise to add to the data
        """
        self.n_points = n_points
        self.x_range = x_range
        self.y_range = y_range
        self.noise_level = noise_level
        
        # Generate initial data
        self.generate_data()
        
        # Calculate best fit parameters
        self.best_slope, self.best_intercept = self.calculate_best_fit()
        
        # Current line parameters
        self.current_slope = self.best_slope
        self.current_intercept = self.best_intercept
        
    def generate_data(self):
        """Generate random data points with noise."""
        np.random.seed(42)  # For reproducible results
        
        # Generate x values
        self.x = np.random.uniform(self.x_range[0], self.x_range[1], self.n_points)
        
        # Generate y values with underlying trend and noise
        base_slope = 0.25
        base_intercept = 4
        noise = np.random.normal(0, self.noise_level, self.n_points)
        self.y = base_intercept + base_slope * self.x + noise
        
        # Clamp y values to specified range
        self.y = np.clip(self.y, self.y_range[0], self.y_range[1])
        
        # Sort by x for better visualization
        sorted_indices = np.argsort(self.x)
        self.x = self.x[sorted_indices]
        self.y = self.y[sorted_indices]
        
        # Calculate mean of y
        self.mean_y = np.mean(self.y)
        
    def calculate_best_fit(self):
        """Calculate the best fit line parameters using least squares."""
        mean_x = np.mean(self.x)
        
        numerator = np.sum((self.x - mean_x) * (self.y - self.mean_y))
        denominator = np.sum((self.x - mean_x) ** 2)
        
        slope = numerator / denominator
        intercept = self.mean_y - slope * mean_x
        
        return slope, intercept
    
    def calculate_r_squared(self, slope, intercept):
        """Calculate R² for given line parameters."""
        y_pred = slope * self.x + intercept
        
        ss_res = np.sum((self.y - y_pred) ** 2)
        ss_tot = np.sum((self.y - self.mean_y) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot)
        mse_regression = ss_res / self.n_points
        mse_mean = ss_tot / self.n_points
        
        return r_squared, mse_regression, mse_mean
    
    def create_plots(self, slope, intercept):
        """Create the side-by-side plots."""
        # Calculate predictions and statistics
        y_pred = slope * self.x + intercept
        r_squared, mse_regression, mse_mean = self.calculate_r_squared(slope, intercept)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Regression Line', 'Mean Line'),
            horizontal_spacing=0.1
        )
        
        # Left plot: Regression line
        # Regression line
        x_line = np.array([self.x_range[0], self.x_range[1]])
        y_line = slope * x_line + intercept
        
        fig.add_trace(
            go.Scatter(x=x_line, y=y_line, mode='lines', name='Regression Line',
                      line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Data points
        fig.add_trace(
            go.Scatter(x=self.x, y=self.y, mode='markers', name='Data Points',
                      marker=dict(color='blue', size=8)),
            row=1, col=1
        )
        
        # Residual lines
        for i in range(self.n_points):
            fig.add_trace(
                go.Scatter(x=[self.x[i], self.x[i]], y=[self.y[i], y_pred[i]],
                          mode='lines', line=dict(color='red', width=1.5, dash='dash'),
                          showlegend=False, hoverinfo='skip'),
                row=1, col=1
            )
        
        # Right plot: Mean line
        # Mean line
        fig.add_trace(
            go.Scatter(x=x_line, y=[self.mean_y, self.mean_y], mode='lines', 
                      name='Mean Line', line=dict(color='orange', width=3)),
            row=1, col=2
        )
        
        # Data points
        fig.add_trace(
            go.Scatter(x=self.x, y=self.y, mode='markers', name='Data Points',
                      marker=dict(color='blue', size=8), showlegend=False),
            row=1, col=2
        )
        
        # Deviation lines
        for i in range(self.n_points):
            fig.add_trace(
                go.Scatter(x=[self.x[i], self.x[i]], y=[self.y[i], self.mean_y],
                          mode='lines', line=dict(color='orange', width=1.5, dash='dash'),
                          showlegend=False, hoverinfo='skip'),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f'R² = {r_squared:.3f} | Equation: y = {slope:.2f}x + {intercept:.2f}',
            showlegend=False,
            height=500,
            plot_bgcolor='white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="X", range=[self.x_range[0]-1, self.x_range[1]+1], 
                        showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Y", range=[self.y_range[0]-1, self.y_range[1]+1], 
                        showgrid=True, gridcolor='lightgray')
        
        return fig, r_squared, mse_regression, mse_mean
    
    def display_statistics(self, r_squared, mse_regression, mse_mean, slope, intercept):
        """Display statistics below the plots."""
        improvement_percent = ((mse_mean - mse_regression) / mse_mean * 100) if mse_regression < mse_mean else 0
        
        stats_html = f"""
        <div style='background: #e8f4fd; padding: 20px; border-radius: 8px; text-align: center; margin-top: 20px;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>Model Performance Statistics</h3>
            <div style='display: flex; justify-content: space-around; margin: 15px 0;'>
                <div style='text-align: center;'>
                    <div style='font-size: 1.4em; font-weight: bold; color: #3498db;'>{r_squared:.3f}</div>
                    <div style='font-size: 0.9em; color: #6c757d;'>R² Value</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 1.4em; font-weight: bold; color: #3498db;'>{improvement_percent:.1f}%</div>
                    <div style='font-size: 0.9em; color: #6c757d;'>Error Reduction</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 1.4em; font-weight: bold; color: #3498db;'>{mse_regression:.2f}</div>
                    <div style='font-size: 0.9em; color: #6c757d;'>MSE (Regression)</div>
                </div>
                <div style='text-align: center;'>
                    <div style='font-size: 1.4em; font-weight: bold; color: #3498db;'>{mse_mean:.2f}</div>
                    <div style='font-size: 0.9em; color: #6c757d;'>MSE (Mean)</div>
                </div>
            </div>
        </div>
        """
        return HTML(stats_html)
    
    def interactive_plot(self):
        """Create interactive plot with sliders."""
        def update_plot(slope, intercept):
            fig, r_squared, mse_regression, mse_mean = self.create_plots(slope, intercept)
            fig.show()
            stats = self.display_statistics(r_squared, mse_regression, mse_mean, slope, intercept)
            display.display(stats)
        
        # Create sliders
        slope_slider = FloatSlider(
            value=self.best_slope,
            min=-1.0,
            max=1.0,
            step=0.05,
            description='Slope:',
            style={'description_width': 'initial'}
        )
        
        intercept_slider = FloatSlider(
            value=self.best_intercept,
            min=0.0,
            max=10.0,
            step=0.1,
            description='Intercept:',
            style={'description_width': 'initial'}
        )
        
        interact(update_plot, slope=slope_slider, intercept=intercept_slider)
    
    def static_plot(self, slope=None, intercept=None):
        """Create a static plot with specified or best-fit parameters."""
        if slope is None:
            slope = self.best_slope
        if intercept is None:
            intercept = self.best_intercept
            
        fig, r_squared, mse_regression, mse_mean = self.create_plots(slope, intercept)
        fig.show()
        
        stats = self.display_statistics(r_squared, mse_regression, mse_mean, slope, intercept)
        display.display(stats)
        
        return fig
    
    def compare_lines(self, slopes, intercepts, titles=None):
        """Compare multiple regression lines."""
        if titles is None:
            titles = [f'Line {i+1}' for i in range(len(slopes))]
        
        fig = make_subplots(
            rows=1, cols=len(slopes),
            subplot_titles=titles,
            horizontal_spacing=0.05
        )
        
        for i, (slope, intercept) in enumerate(zip(slopes, intercepts)):
            y_pred = slope * self.x + intercept
            r_squared, _, _ = self.calculate_r_squared(slope, intercept)
            
            # Regression line
            x_line = np.array([self.x_range[0], self.x_range[1]])
            y_line = slope * x_line + intercept
            
            fig.add_trace(
                go.Scatter(x=x_line, y=y_line, mode='lines', name=f'Line {i+1}',
                          line=dict(color='red', width=3)),
                row=1, col=i+1
            )
            
            # Data points
            fig.add_trace(
                go.Scatter(x=self.x, y=self.y, mode='markers', 
                          marker=dict(color='blue', size=6), showlegend=False),
                row=1, col=i+1
            )
            
            # Residual lines
            for j in range(self.n_points):
                fig.add_trace(
                    go.Scatter(x=[self.x[j], self.x[j]], y=[self.y[j], y_pred[j]],
                              mode='lines', line=dict(color='red', width=1, dash='dash'),
                              showlegend=False, hoverinfo='skip'),
                    row=1, col=i+1
                )
            
            # Update subplot title with R²
            fig.layout.annotations[i].text = f'{titles[i]}<br>R² = {r_squared:.3f}'
        
        fig.update_layout(
            title='Comparing Different Regression Lines',
            showlegend=False,
            height=400,
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(title_text="X", range=[self.x_range[0]-1, self.x_range[1]+1])
        fig.update_yaxes(title_text="Y", range=[self.y_range[0]-1, self.y_range[1]+1])
        
        fig.show()
        return fig

# Convenience functions for easy use
def create_r_squared_demo(n_points=25, interactive=True):
    """
    Create an R² demonstration.
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    interactive : bool
        If True, creates interactive sliders. If False, shows static best-fit plot.
    
    Returns:
    --------
    RSquaredVisualizer object
    """
    viz = RSquaredVisualizer(n_points=n_points)
    
    if interactive:
        print("🎯 Interactive R² Demonstration")
        print("Use the sliders below to adjust the regression line and see how R² changes!")
        print("Higher R² = better fit = smaller residuals")
        print("-" * 60)
        viz.interactive_plot()
    else:
        viz.static_plot()
    
    return viz

def compare_fits_demo():
    """
    Demonstrate the effect of different line parameters on R².
    """
    viz = RSquaredVisualizer(n_points=25)
    
    # Compare best fit vs poor fits
    slopes = [viz.best_slope, 0.1, 0.5, -0.2]
    intercepts = [viz.best_intercept, 6, 3, 8]
    titles = ['Best Fit', 'Too Flat', 'Too Steep', 'Wrong Direction']
    
    print("📊 Comparing Different Regression Lines")
    print("Notice how R² decreases as the line fits the data more poorly!")
    print("-" * 60)
    
    viz.compare_lines(slopes, intercepts, titles)
    return viz

# Example usage functions
def example_usage():
    """
    Show example usage of the R² visualizer.
    """
    print("=" * 60)
    print("R² VISUALIZER - EXAMPLE USAGE")
    print("=" * 60)
    
    print("\n1. Interactive Demo:")
    print("   viz = create_r_squared_demo(interactive=True)")
    
    print("\n2. Static Plot:")
    print("   viz = create_r_squared_demo(interactive=False)")
    
    print("\n3. Compare Different Lines:")
    print("   compare_fits_demo()")
    
    print("\n4. Custom Usage:")
    print("   viz = RSquaredVisualizer(n_points=30)")
    print("   viz.static_plot(slope=0.3, intercept=5)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    example_usage()