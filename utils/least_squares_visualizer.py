"""
Combined Least Squares Visualizers
Interactive demonstrations of the least squares method for both synthetic and real data
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, FloatSlider, Button, HBox, VBox, Output, Tab
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.linear_model import LinearRegression

class BaseLeastSquaresVisualizer:
    """Base class with common functionality for least squares visualizers"""
    
    def calculate_rss(self, slope, intercept):
        """Calculate Sum of Squared Errors"""
        predicted = slope * self.x_data + intercept
        residuals = self.y_data - predicted
        return np.sum(residuals ** 2)
    
    def calculate_r_squared(self, slope, intercept):
        """Calculate R-squared coefficient of determination"""
        predicted = slope * self.x_data + intercept
        ss_res = np.sum((self.y_data - predicted) ** 2)
        ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def calculate_optimal_line_formula(self):
        """Calculate optimal slope and intercept using least squares formula"""
        n = len(self.x_data)
        x_mean = np.mean(self.x_data)
        y_mean = np.mean(self.y_data)
        
        # Calculate slope using the least squares formula
        numerator = np.sum((self.x_data - x_mean) * (self.y_data - y_mean))
        denominator = np.sum((self.x_data - x_mean) ** 2)
        slope = numerator / denominator
        
        # Calculate intercept
        intercept = y_mean - slope * x_mean
        
        return slope, intercept

# class GeneralLeastSquaresVisualizer(BaseLeastSquaresVisualizer):
    """General least squares visualizer with synthetic data"""
    
    def __init__(self):
        # Sample data points - more data points with some noise
        np.random.seed(42)  # For reproducible results
        self.x_data = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9])
        # Create y data with trend + noise
        true_slope = 1.8
        true_intercept = 0.5
        noise = np.random.normal(0, 0.8, len(self.x_data))
        self.y_data = true_slope * self.x_data + true_intercept + noise
        
        # Calculate optimal parameters using least squares
        self.optimal_slope, self.optimal_intercept = self.calculate_optimal_line_formula()
        
        # Current parameters
        self.current_slope = 2.0
        self.current_intercept = 1.0
        
        # Flag to show optimal line
        self.show_optimal = False
        
        # Create widgets
        self.setup_widgets()
        
        # Output widget for plots
        self.output = Output()
    
    def setup_widgets(self):
        """Create interactive widgets"""
        self.slope_slider = FloatSlider(
            value=2.0,
            min=-1.0,
            max=4.0,
            step=0.1,
            description='Slope (β₁):',
            style={'description_width': 'initial'},
            continuous_update=True
        )
        
        self.intercept_slider = FloatSlider(
            value=1.0,
            min=-3.0,
            max=5.0,
            step=0.1,
            description='Intercept (β₀):',
            style={'description_width': 'initial'},
            continuous_update=True
        )
        
        self.rss_label = widgets.HTML(
            value=f"<b style='color: #1976d2; font-size: 16px; padding: 10px; background: #e3f2fd; border-radius: 5px;'>SSE: {self.calculate_rss(2.0, 1.0):.2f}</b>"
        )
        
        self.optimal_button = Button(
            description='Find Optimal Line',
            button_style='success',
            tooltip='Jump to the mathematically optimal solution',
            layout=widgets.Layout(height='40px', width='150px')
        )
        
        # Set up event handlers
        self.slope_slider.observe(self.on_parameter_change, names='value')
        self.intercept_slider.observe(self.on_parameter_change, names='value')
        self.optimal_button.on_click(self.find_optimal)
    
    def on_parameter_change(self, change):
        """Handle parameter changes"""
        self.current_slope = self.slope_slider.value
        self.current_intercept = self.intercept_slider.value
        self.update_plot()
    
    def find_optimal(self, button):
        """Set parameters to optimal values and show optimal line"""
        self.slope_slider.value = self.optimal_slope
        self.intercept_slider.value = self.optimal_intercept
        self.current_slope = self.optimal_slope
        self.current_intercept = self.optimal_intercept
        self.show_optimal = True
        self.update_plot()
    
    def create_residual_shapes(self):
        """Create shapes to visualize squared residuals"""
        shapes = []
        predicted = self.current_slope * self.x_data + self.current_intercept
        
        for i, (x, y_actual, y_pred) in enumerate(zip(self.x_data, self.y_data, predicted)):
            residual = y_actual - y_pred
            square_size = abs(residual) * 0.3
            
            # Create small squares to represent squared residuals
            shapes.append(
                dict(
                    type="rect",
                    x0=x - square_size/2,
                    x1=x + square_size/2,
                    y0=min(y_actual, y_pred),
                    y1=min(y_actual, y_pred) + square_size,
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(color="red", width=1)
                )
            )
        
        return shapes
    
    def update_plot(self):
        """Update the main plot"""
        with self.output:
            clear_output(wait=True)
            
            # Calculate current RSS
            current_rss = self.calculate_rss(self.current_slope, self.current_intercept)
            
            # Update RSS label
            self.rss_label.value = f"<b style='color: #1976d2; font-size: 16px; padding: 10px; background: #e3f2fd; border-radius: 5px;'>SSE: {current_rss:.2f}</b>"
            
            # Create the plot
            fig = go.Figure()
            
            # Generate line data
            x_line = np.linspace(0, 9, 100)
            y_current = self.current_slope * x_line + self.current_intercept
            y_optimal = self.optimal_slope * x_line + self.optimal_intercept
            predicted_points = self.current_slope * self.x_data + self.current_intercept
            
            # Data points
            fig.add_trace(
                go.Scatter(
                    x=self.x_data,
                    y=self.y_data,
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='#2E86C1', size=12, line=dict(color='white', width=2)),
                    hovertemplate='<b>Point (%{x}, %{y})</b><extra></extra>'
                )
            )
            
            # Current line
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_current,
                    mode='lines',
                    name=f'Current Line: y = {self.current_slope:.2f}x + {self.current_intercept:.2f}',
                    line=dict(color='#E67E22', width=4),
                    hovertemplate='<b>Current Line</b><br>y = %{y:.2f}<extra></extra>'
                )
            )
            
            # Optimal line (only show if button was clicked)
            if self.show_optimal:
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_optimal,
                        mode='lines',
                        name=f'Optimal Line: y = {self.optimal_slope:.2f}x + {self.optimal_intercept:.2f}',
                        line=dict(color='#27AE60', width=3, dash='dash'),
                        hovertemplate='<b>Optimal Line</b><br>y = %{y:.2f}<extra></extra>'
                    )
                )
            
            # Residual lines (errors)
            for i, (x, y_actual, y_pred) in enumerate(zip(self.x_data, self.y_data, predicted_points)):
                residual = y_actual - y_pred
                fig.add_trace(
                    go.Scatter(
                        x=[x, x],
                        y=[y_actual, y_pred],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dot'),
                        showlegend=False,
                        hovertemplate=f'<b>Residual</b><br>Error: {residual:.2f}<br>Squared: {residual**2:.2f}<extra></extra>',
                        name=f'Residual {i+1}'
                    )
                )
            
            # Update layout with shapes for squared residuals
            fig.update_layout(
                title=dict(
                    text="<b>General Least Squares Method: Minimize Sum of Squared Residuals</b>",
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title="<b>x (Explanatory Variable)</b>",
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[-0.5, 9.5]
                ),
                yaxis=dict(
                    title="<b>y (Response Variable)</b>",
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                plot_bgcolor='rgba(240,240,240,0.1)',
                paper_bgcolor='white',
                height=600,
                width=900,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='lightgray',
                    borderwidth=1
                ),
                shapes=self.create_residual_shapes()
            )
            
            fig.show()
    
    def display(self):
        """Display the complete interactive visualization"""
        # Create the layout
        controls = HBox([
            self.slope_slider,
            self.intercept_slider,
            self.rss_label,
            self.optimal_button
        ])
        
        # Instructions
        instructions = widgets.HTML(
            value="""
            <div style='background: #f0f7ff; padding: 15px; border-left: 4px solid #2196f3; margin: 10px 0; border-radius: 5px;'>
                <h3 style='margin-top: 0;'>📊 General Least Squares Method Interactive Demo</h3>
                <b>How it works:</b>
                <ul>
                    <li><b>Adjust sliders</b> to try different lines and see how they fit the data</li>
                    <li><b>Sum of Squared Errors (SSE)</b> measures total error - lower is better!</li>
                    <li><b>Red dotted lines</b> show residuals (prediction errors)</li>
                    <li><b>Red squares</b> visualize the "squares" in "least squares"</li>
                    <li><b>Goal:</b> Try to minimize SSE manually by adjusting the sliders</li>
                    <li><b>Click "Find Optimal Line"</b> to reveal the mathematically perfect solution</li>
                </ul>
            </div>
            """
        )
        
        full_widget = VBox([instructions, controls, self.output])
        display(full_widget)
        
        # Initial plot
        self.update_plot()


class GeneralLeastSquaresVisualizer(BaseLeastSquaresVisualizer):
    """General least squares visualizer with synthetic data"""
    
    def __init__(self):
        # Sample data points - more data points with some noise
        np.random.seed(42)  # For reproducible results
        self.x_data = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9])
        # Create y data with trend + noise
        true_slope = 1.8
        true_intercept = 0.5
        noise = np.random.normal(0, 0.8, len(self.x_data))
        self.y_data = true_slope * self.x_data + true_intercept + noise
        
        # Calculate optimal parameters using least squares
        self.optimal_slope, self.optimal_intercept = self.calculate_optimal_line_formula()
        
        # Current parameters
        self.current_slope = 2.0
        self.current_intercept = 1.0
        
        # Flag to show optimal line
        self.show_optimal = False
        
        # Create widgets
        self.setup_widgets()
        
        # Output widget for plots
        self.output = Output()
    
    def setup_widgets(self):
        """Create interactive widgets"""
        self.slope_slider = FloatSlider(
            value=2.0,
            min=-1.0,
            max=4.0,
            step=0.1,
            description='Slope (β₁):',
            style={'description_width': 'initial'},
            continuous_update=True
        )
        
        self.intercept_slider = FloatSlider(
            value=1.0,
            min=-3.0,
            max=5.0,
            step=0.1,
            description='Intercept (β₀):',
            style={'description_width': 'initial'},
            continuous_update=True
        )
        
        self.rss_label = widgets.HTML(
            value=f"<b style='color: #1976d2; font-size: 16px; padding: 10px; background: #e3f2fd; border-radius: 5px;'>SSE: {self.calculate_rss(2.0, 1.0):.2f}</b>"
        )
        
        self.optimal_button = Button(
            description='Find Optimal Line',
            button_style='success',
            tooltip='Jump to the mathematically optimal solution',
            layout=widgets.Layout(height='40px', width='150px')
        )
        
        # Set up event handlers
        self.slope_slider.observe(self.on_parameter_change, names='value')
        self.intercept_slider.observe(self.on_parameter_change, names='value')
        self.optimal_button.on_click(self.find_optimal)
    
    def on_parameter_change(self, change):
        """Handle parameter changes"""
        self.current_slope = self.slope_slider.value
        self.current_intercept = self.intercept_slider.value
        self.update_plot()
    
    def find_optimal(self, button):
        """Set parameters to optimal values and show optimal line"""
        self.slope_slider.value = self.optimal_slope
        self.intercept_slider.value = self.optimal_intercept
        self.current_slope = self.optimal_slope
        self.current_intercept = self.optimal_intercept
        self.show_optimal = True
        self.update_plot()
    
    def create_residual_shapes(self):
        """Create shapes to visualize squared residuals"""
        shapes = []
        predicted = self.current_slope * self.x_data + self.current_intercept

        # 1. Define the plot's pixel dimensions (from update_plot)
        plot_width_px = 900
        plot_height_px = 600

        # 2. Define the data ranges for each axis (from update_plot)
        x_range_data = 9.5 - (-0.5)
        y_range_data = (np.max(self.y_data) + 2) - (np.min(self.y_data) - 2)

        # 3. Calculate the scaling factor to convert y-axis data units to x-axis data units
        # such that the resulting shape is a square in pixel space.
        # w_data = h_data * (plot_height_px / plot_width_px) * (x_range_data / y_range_data)
        scaling_factor = (plot_height_px / plot_width_px) * (x_range_data / y_range_data)

        for x, y_actual, y_pred in zip(self.x_data, self.y_data, predicted):
            residual = y_actual - y_pred
            
            # The height of the square in y-axis data units
            square_height_data = abs(residual)

            # The required width in x-axis data units to make the shape a square
            square_width_data = square_height_data * scaling_factor
            
            # Define the square's corners
            x0 = x
            y0 = min(y_actual, y_pred)
            x1 = x0 + square_width_data
            y1 = y0 + square_height_data
            
            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    line=dict(color="red", width=1)
                )
            )
        
        return shapes

    def update_plot(self):
        """Update the main plot"""
        with self.output:
            clear_output(wait=True)
            
            # Calculate current RSS
            current_rss = self.calculate_rss(self.current_slope, self.current_intercept)
            
            # Update RSS label
            self.rss_label.value = f"<b style='color: #1976d2; font-size: 16px; padding: 10px; background: #e3f2fd; border-radius: 5px;'>SSE: {current_rss:.2f}</b>"
            
            # Create the plot
            fig = go.Figure()
            
            # Generate line data
            x_line = np.linspace(0, 9, 100)
            y_current = self.current_slope * x_line + self.current_intercept
            y_optimal = self.optimal_slope * x_line + self.optimal_intercept
            predicted_points = self.current_slope * self.x_data + self.current_intercept
            
            # Data points
            fig.add_trace(
                go.Scatter(
                    x=self.x_data,
                    y=self.y_data,
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='#2E86C1', size=12, line=dict(color='white', width=2)),
                    hovertemplate='<b>Point (%{x}, %{y})</b><extra></extra>'
                )
            )
            
            # Current line
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_current,
                    mode='lines',
                    name=f'Current Line: y = {self.current_slope:.2f}x + {self.current_intercept:.2f}',
                    line=dict(color='#E67E22', width=4),
                    hovertemplate='<b>Current Line</b><br>y = %{y:.2f}<extra></extra>'
                )
            )
            
            # Optimal line (only show if button was clicked)
            if self.show_optimal:
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_optimal,
                        mode='lines',
                        name=f'Optimal Line: y = {self.optimal_slope:.2f}x + {self.optimal_intercept:.2f}',
                        line=dict(color='#27AE60', width=3, dash='dash'),
                        hovertemplate='<b>Optimal Line</b><br>y = %{y:.2f}<extra></extra>'
                    )
                )
            
            # Residual lines (errors)
            for i, (x, y_actual, y_pred) in enumerate(zip(self.x_data, self.y_data, predicted_points)):
                residual = y_actual - y_pred
                fig.add_trace(
                    go.Scatter(
                        x=[x, x],
                        y=[y_actual, y_pred],
                        mode='lines',
                        line=dict(color='red', width=2, dash='dot'),
                        showlegend=False,
                        hovertemplate=f'<b>Residual</b><br>Error: {residual:.2f}<br>Squared: {residual**2:.2f}<extra></extra>',
                        name=f'Residual {i+1}'
                    )
                )
            
            # Update layout with shapes for squared residuals
            fig.update_layout(
                title=dict(
                    text="<b>General Least Squares Method: Minimize Sum of Squared Residuals</b>",
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis=dict(
                    title="<b>x (Explanatory Variable)</b>",
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[-0.5, 9.5]
                ),
                yaxis=dict(
                    title="<b>y (Response Variable)</b>",
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[np.min(self.y_data) - 2, np.max(self.y_data) + 2] # Set explicit y-range
                ),
                plot_bgcolor='rgba(240,240,240,0.1)',
                paper_bgcolor='white',
                height=600,
                width=900,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='lightgray',
                    borderwidth=1
                ),
                shapes=self.create_residual_shapes()
            )
            
            fig.show()
    
    def display(self):
        """Display the complete interactive visualization"""
        # Create the layout
        controls = HBox([
            self.slope_slider,
            self.intercept_slider,
            self.rss_label,
            self.optimal_button
        ])
        
        # Instructions
        instructions = widgets.HTML(
            value="""
            <div style='background: #f0f7ff; padding: 15px; border-left: 4px solid #2196f3; margin: 10px 0; border-radius: 5px;'>
                <h3 style='margin-top: 0;'>📊 General Least Squares Method Interactive Demo</h3>
                <b>How it works:</b>
                <ul>
                    <li><b>Adjust sliders</b> to try different lines and see how they fit the data</li>
                    <li><b>Sum of Squared Errors (SSE)</b> measures total error - lower is better!</li>
                    <li><b>Red dotted lines</b> show residuals (prediction errors)</li>
                    <li><b>Red squares</b> visualize the "squares" in "least squares"</li>
                    <li><b>Goal:</b> Try to minimize SSE manually by adjusting the sliders</li>
                    <li><b>Click "Find Optimal Line"</b> to reveal the mathematically perfect solution</li>
                </ul>
            </div>
            """
        )
        
        full_widget = VBox([instructions, controls, self.output])
        display(full_widget)
        
        # Initial plot
        self.update_plot()


class TVSalesVisualizer(BaseLeastSquaresVisualizer):
    """TV Sales specific least squares visualizer using real advertising data"""
    
    def __init__(self, df_tv_sales):
        # Use actual TV Sales data
        self.x_data = df_tv_sales['TV'].values
        self.y_data = df_tv_sales['sales'].values
        
        # Calculate optimal parameters using sklearn (matches notebook)
        model = LinearRegression()
        model.fit(self.x_data.reshape(-1, 1), self.y_data)
        self.optimal_slope = model.coef_[0]
        self.optimal_intercept = model.intercept_
        
        # Calculate optimal using manual least squares formula for verification
        self.manual_slope, self.manual_intercept = self.calculate_optimal_line_formula()
        
        # Current parameters - start with reasonable guess
        self.current_slope = 0.05
        self.current_intercept = 5.0
        
        # Flag to show optimal line
        self.show_optimal = False
        
        # Create widgets
        self.setup_widgets()
        
        # Output widget for plots
        self.output = Output()
    
    def setup_widgets(self):
        """Create interactive widgets"""
        self.slope_slider = FloatSlider(
            value=0.05,
            min=-0.01,
            max=0.1,
            step=0.001,
            description='Slope (β₁):',
            style={'description_width': 'initial'},
            continuous_update=True,
            readout_format='.4f'
        )
        
        self.intercept_slider = FloatSlider(
            value=5.0,
            min=0.0,
            max=15.0,
            step=0.1,
            description='Intercept (β₀):',
            style={'description_width': 'initial'},
            continuous_update=True
        )
        
        # Initialize RSS and R² labels
        initial_rss = self.calculate_rss(0.05, 5.0)
        initial_r2 = self.calculate_r_squared(0.05, 5.0)
        
        self.rss_label = widgets.HTML(
            value=f"<b style='color: #d32f2f; font-size: 14px; padding: 8px; background: #ffebee; border-radius: 5px;'>RSS: {initial_rss:.2f}</b>"
        )
        
        self.r2_label = widgets.HTML(
            value=f"<b style='color: #1976d2; font-size: 14px; padding: 8px; background: #e3f2fd; border-radius: 5px;'>R²: {initial_r2:.4f}</b>"
        )
        
        self.optimal_button = Button(
            description='Show Optimal Line',
            button_style='success',
            tooltip='Reveal the mathematically optimal solution',
            layout=widgets.Layout(height='40px', width='160px')
        )
        
        self.info_label = widgets.HTML(
            value="<b style='color: #666; font-size: 12px;'>Try to minimize SSE and maximize R²!</b>"
        )
        
        # Set up event handlers
        self.slope_slider.observe(self.on_parameter_change, names='value')
        self.intercept_slider.observe(self.on_parameter_change, names='value')
        self.optimal_button.on_click(self.find_optimal)
    
    def on_parameter_change(self, change):
        """Handle parameter changes"""
        self.current_slope = self.slope_slider.value
        self.current_intercept = self.intercept_slider.value
        self.update_plot()
    
    def find_optimal(self, button):
        """Set parameters to optimal values and show optimal line"""
        self.slope_slider.value = self.optimal_slope
        self.intercept_slider.value = self.optimal_intercept
        self.current_slope = self.optimal_slope
        self.current_intercept = self.optimal_intercept
        self.show_optimal = True
        self.optimal_button.description = 'Optimal Line Shown'
        self.optimal_button.button_style = 'info'
        self.update_plot()
    
    def create_residual_shapes(self):
        """Create shapes to visualize squared residuals"""
        shapes = []
        predicted = self.current_slope * self.x_data + self.current_intercept
        
        # Only show a sample of residual squares to avoid cluttering
        indices = np.random.choice(len(self.x_data), size=min(15, len(self.x_data)), replace=False)
        
        for i in indices:
            x, y_actual, y_pred = self.x_data[i], self.y_data[i], predicted[i]
            residual = y_actual - y_pred
            square_size = abs(residual) * 0.8  # Scale factor for visibility
            
            # Create small squares to represent squared residuals
            shapes.append(
                dict(
                    type="rect",
                    x0=x - square_size/2,
                    x1=x + square_size/2,
                    y0=min(y_actual, y_pred),
                    y1=min(y_actual, y_pred) + square_size,
                    fillcolor="rgba(255, 99, 71, 0.2)",
                    line=dict(color="rgba(255, 99, 71, 0.6)", width=1)
                )
            )
        
        return shapes
    
    def update_plot(self):
        """Update the main plot"""
        with self.output:
            clear_output(wait=True)
            
            # Calculate current metrics
            current_rss = self.calculate_rss(self.current_slope, self.current_intercept)
            current_r2 = self.calculate_r_squared(self.current_slope, self.current_intercept)
            
            # Update metric labels
            self.rss_label.value = f"<b style='color: #d32f2f; font-size: 14px; padding: 8px; background: #ffebee; border-radius: 5px;'>SSE: {current_rss:.2f}</b>"
            self.r2_label.value = f"<b style='color: #1976d2; font-size: 14px; padding: 8px; background: #e3f2fd; border-radius: 5px;'>R²: {current_r2:.4f}</b>"
            
            # Create the plot
            fig = go.Figure()
            
            # Generate line data
            x_line = np.linspace(self.x_data.min() - 10, self.x_data.max() + 10, 100)
            y_current = self.current_slope * x_line + self.current_intercept
            y_optimal = self.optimal_slope * x_line + self.optimal_intercept
            predicted_points = self.current_slope * self.x_data + self.current_intercept
            
            # Data points
            fig.add_trace(
                go.Scatter(
                    x=self.x_data,
                    y=self.y_data,
                    mode='markers',
                    name='TV vs Sales Data',
                    marker=dict(
                        color='#2E86C1', 
                        size=8, 
                        line=dict(color='white', width=1),
                        opacity=0.8
                    ),
                    hovertemplate='<b>TV: $%{x:.0f}</b><br><b>Sales: %{y:.1f}</b><extra></extra>'
                )
            )
            
            # Current line
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_current,
                    mode='lines',
                    name=f'Your Line: Sales = {self.current_intercept:.2f} + {self.current_slope:.4f} × TV',
                    line=dict(color='#E67E22', width=4),
                    hovertemplate='<b>Your Prediction</b><br>Sales = %{y:.2f}<extra></extra>'
                )
            )
            
            # Optimal line (only show if button was clicked)
            if self.show_optimal:
                optimal_rss = self.calculate_rss(self.optimal_slope, self.optimal_intercept)
                optimal_r2 = self.calculate_r_squared(self.optimal_slope, self.optimal_intercept)
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_optimal,
                        mode='lines',
                        name=f'Optimal Line: Sales = {self.optimal_intercept:.2f} + {self.optimal_slope:.4f} × TV',
                        line=dict(color='#27AE60', width=3, dash='dash'),
                        hovertemplate=f'<b>Optimal Line</b><br>Sales = %{{y:.2f}}<br>RSS: {optimal_rss:.2f}<br>R²: {optimal_r2:.4f}<extra></extra>'
                    )
                )
            
            # Residual lines (show subset to avoid cluttering)
            sample_indices = np.random.choice(len(self.x_data), size=min(25, len(self.x_data)), replace=False)
            for i in sample_indices:
                x, y_actual, y_pred = self.x_data[i], self.y_data[i], predicted_points[i]
                residual = y_actual - y_pred
                fig.add_trace(
                    go.Scatter(
                        x=[x, x],
                        y=[y_actual, y_pred],
                        mode='lines',
                        line=dict(color='rgba(255, 99, 71, 0.6)', width=1.5),
                        showlegend=False,
                        hovertemplate=f'<b>Residual</b><br>TV: ${x:.0f}<br>Actual: {y_actual:.1f}<br>Predicted: {y_pred:.2f}<br>Error: {residual:.2f}<br>Squared Error: {residual**2:.2f}<extra></extra>',
                        name=f'Residual'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="<b>TV Advertising vs Sales: Find the Best Fitting Line!</b><br><span style='font-size: 14px; color: #666;'>Minimize SSE (Sum of Squared Errors) and Maximize R²</span>",
                    x=0.5,
                    font=dict(size=16)
                ),
                xaxis=dict(
                    title="<b>TV Advertising Budget ($)</b>",
                    showgrid=True,
                    gridcolor='lightgray',
                    range=[self.x_data.min() - 10, self.x_data.max() + 10]
                ),
                yaxis=dict(
                    title="<b>Sales (in thousands of units)</b>",
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                plot_bgcolor='rgba(240,240,240,0.05)',
                paper_bgcolor='white',
                height=650,
                width=1000,
                showlegend=True,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='lightgray',
                    borderwidth=1
                ),
                shapes=self.create_residual_shapes()
            )
            
            fig.show()
    
    def display(self):
        """Display the complete interactive visualization"""
        # Create the layout
        controls_row1 = HBox([self.slope_slider, self.intercept_slider])
        controls_row2 = HBox([self.rss_label, self.r2_label, self.optimal_button])
        
        # Instructions specific to TV Sales data
        instructions = widgets.HTML(
            value=f"""
            <div style='background: #f0f7ff; padding: 15px; border-left: 4px solid #2196f3; margin: 10px 0; border-radius: 5px;'>
                <h3 style='margin-top: 0;'>📺 TV Advertising vs Sales: Interactive Least Squares Demo</h3>
                <b>Dataset:</b> Real advertising data with {len(self.x_data)} markets showing TV budget vs Sales
                <br><br>
                <b>Your Mission:</b>
                <ul>
                    <li><b>Find the best line</b> that predicts Sales based on TV advertising budget</li>
                    <li><b>Minimize SSE</b> (Sum of Squared Errors) - the total squared error</li>
                    <li><b>Maximize R²</b> (coefficient of determination) - how well the line explains the data</li>
                    <li><b>Interpretation:</b> Slope = additional sales per $1 spent on TV ads</li>
                    <li><b>Red lines</b> show prediction errors for each market</li>
                    <li><b>Red squares</b> visualize the "squared" part of least squares</li>
                </ul>
                <b>Try adjusting the sliders manually first, then click "Show Optimal Line" to see the solution!</b>
            </div>
            """
        )
        
        summary_widget = widgets.HTML(
            value=f"""
            <div style='background: #fff3e0; padding: 10px; border-left: 4px solid #ff9800; margin: 10px 0; border-radius: 5px;'>
                <b>📊 Dataset Summary:</b> TV Budget range: ${self.x_data.min():.0f} - ${self.x_data.max():.0f} | 
                Sales range: {self.y_data.min():.1f} - {self.y_data.max():.1f} thousand units |
                Optimal Solution: SSE = {self.calculate_rss(self.optimal_slope, self.optimal_intercept):.2f}, 
                R² = {self.calculate_r_squared(self.optimal_slope, self.optimal_intercept):.4f}
            </div>
            """
        )
        
        full_widget = VBox([
            instructions, 
            summary_widget,
            controls_row1, 
            controls_row2,
            self.info_label,
            self.output
        ])
        display(full_widget)
        
        # Initial plot
        self.update_plot()

class CombinedLeastSquaresVisualizer:
    """Combined visualizer with tabs for both general and TV sales demos"""
    
    def __init__(self, df_tv_sales=None):
        self.df_tv_sales = df_tv_sales
        self.general_visualizer = GeneralLeastSquaresVisualizer()
        
        if df_tv_sales is not None:
            self.tv_visualizer = TVSalesVisualizer(df_tv_sales)
        else:
            self.tv_visualizer = None
    
    def display(self):
        """Display both visualizers in tabs"""
        if self.tv_visualizer is not None:
            # Create tabs for both visualizers
            tab_contents = []
            tab_titles = []
            
            # General visualizer tab
            general_output = Output()
            with general_output:
                self.general_visualizer.display()
            tab_contents.append(general_output)
            tab_titles.append("General Demo")
            
            # TV Sales visualizer tab
            tv_output = Output()
            with tv_output:
                self.tv_visualizer.display()
            tab_contents.append(tv_output)
            tab_titles.append("TV Sales Demo")
            
            # Create and display tabs
            tab = Tab(children=tab_contents)
            for i, title in enumerate(tab_titles):
                tab.set_title(i, title)
            
            # Main header
            header = widgets.HTML(
                value="""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; color: white;'>
                    <h1 style='margin: 0; font-size: 24px;'>🎯 Interactive Least Squares Method Visualizers</h1>
                    <p style='margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;'>Learn how the least squares method works with both synthetic and real data</p>
                </div>
                """
            )
            
            display(VBox([header, tab]))
        else:
            # Only show general visualizer
            header = widgets.HTML(
                value="""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; color: white;'>
                    <h1 style='margin: 0; font-size: 24px;'>🎯 Least Squares Method Visualizer</h1>
                    <p style='margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;'>Learn how the least squares method works</p>
                </div>
                """
            )
            display(header)
            self.general_visualizer.display()

# Convenience functions for easy usage
def run_general_visualizer():
    """Create and display the general least squares visualizer"""
    visualizer = GeneralLeastSquaresVisualizer()
    visualizer.display()
    return visualizer

def run_tv_sales_visualizer(df_tv_sales):
    """Create and display the TV Sales visualizer"""
    visualizer = TVSalesVisualizer(df_tv_sales)
    visualizer.display()
    return visualizer

def run_combined_visualizer(df_tv_sales=None):
    """Create and display the combined visualizer with tabs"""
    visualizer = CombinedLeastSquaresVisualizer(df_tv_sales)
    visualizer.display()
    return visualizer

# Backward compatibility functions
def run_visualizer():
    """Backward compatibility: runs the general visualizer"""
    return run_general_visualizer()

def create_tv_sales_visualizer(df_tv_sales):
    """Backward compatibility: runs the TV sales visualizer"""
    return run_tv_sales_visualizer(df_tv_sales)

# Main execution
if __name__ == "__main__":
    print("Combined Least Squares Visualizers")
    print("Available functions:")
    print("- run_general_visualizer(): General demo with synthetic data")
    print("- run_tv_sales_visualizer(df_tv_sales): TV sales demo with real data")
    print("- run_combined_visualizer(df_tv_sales): Both demos in tabs")
