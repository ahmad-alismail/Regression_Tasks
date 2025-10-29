import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

class RegressionAssumptionVisualizer:
    """
    A class to visualize regression assumptions with interactive Plotly plots.
    Supports linearity, homoscedasticity, and autocorrelation assumption visualization.
    """
    
    def __init__(self):
        self.colors = {
            'good': '#2E8B57',      # Sea Green
            'bad': '#DC143C',       # Crimson
            'neutral': '#4682B4',   # Steel Blue
            'background': '#F8F9FA'
        }
    
    def _generate_linear_data(self, n_points=100, noise_level=5):
        """Generate linear relationship data (Good example)"""
        np.random.seed(42)
        pizzas = np.linspace(0, 10, n_points)
        # Linear relationship: happiness = 2 * pizzas + noise
        happiness = 2 * pizzas + 10 + np.random.normal(0, noise_level, n_points)
        return pizzas, happiness
    
    def _generate_nonlinear_data(self, n_points=100, noise_level=3):
        """Generate non-linear relationship data (Bad example)"""
        np.random.seed(42)
        volume = np.linspace(0, 10, n_points)
        # Non-linear relationship: inverted U-shape (quadratic)
        happiness = -0.8 * (volume - 5)**2 + 30 + np.random.normal(0, noise_level, n_points)
        return volume, happiness
    
    def _generate_homoscedastic_data(self, n_points=100, noise_level=800):
        """Generate homoscedastic relationship data (Good example)"""
        np.random.seed(42)
        income = np.linspace(30000, 120000, n_points)  # Income from $30k to $120k
        # Linear relationship: food_spending = 0.05 * income + constant noise
        food_spending = 0.065 * income + 500 + np.random.normal(0, noise_level, n_points)
        return income, food_spending
    
    def _generate_heteroscedastic_data(self, n_points=100):
        """Generate heteroscedastic relationship data (Bad example)"""
        np.random.seed(42)
        income = np.linspace(30000, 120000, n_points)  # Income from $30k to $120k
        
        # Varying noise level - increases with income (funnel/cone shape)
        # Low income: small variance, High income: large variance
        noise_levels = 200 + (income - 30000) * 0.025  # Noise increases with income
        
        # Linear relationship with heteroscedastic errors
        food_spending_base = 0.065 * income + 500
        noise = np.array([np.random.normal(0, noise_level) for noise_level in noise_levels])
        food_spending = food_spending_base + noise
        
        return income, food_spending

    def _generate_independent_errors_data(self, n_points=100, noise_level=15):
        """Generate data with independent errors (Good example)"""
        np.random.seed(42)
        time = np.arange(n_points)
        # Some predictor with a trend
        predictor = 2 * time + np.random.normal(0, 20, n_points)
        # Target variable with independent random noise
        target = 0.5 * predictor + 50 + np.random.normal(0, noise_level, n_points)
        return time, predictor, target

    def _generate_autocorrelated_errors_data(self, n_points=100, noise_level=3):
        """
        Generate data with a clear cyclical pattern in errors (Bad example).
        This version creates an explicit down-up-down pattern.
        """
        np.random.seed(42)
        time = np.arange(n_points)
        
        # A simple linear predictor that the model will use
        predictor = 0.8 * time + np.random.normal(0, 15, n_points)
        
        # Create a strong, clear cyclical pattern (a negative cosine wave)
        # This is the "hidden" seasonal information the linear model won't capture.
        # The frequency is set to show 1.5 cycles (down -> up -> down).
        cyclical_pattern = -40 * np.cos(np.linspace(0, 3 * np.pi, n_points)) 
        
        # The target variable is a combination of the linear trend, 
        # the hidden cyclical pattern, and some random noise.
        target = 1.2 * predictor + 100 + cyclical_pattern + np.random.normal(0, noise_level, n_points)
        
        return time, predictor, target
    
    def _generate_normal_residuals_data(self, n_points=200):
        """
        Generates data for 'Exam Scores vs Hours Studied'.
        This will produce normal residuals (Good example).
        """
        np.random.seed(1)
        hours_studied = np.random.uniform(1, 20, n_points)
        # The true relationship: score increases with hours, but with many small random factors
        noise = np.random.normal(0, 12, n_points) # Symmetrical, normal noise
        exam_score = 40 + 2.5 * hours_studied + noise
        exam_score = np.clip(exam_score, 0, 100) # Scores can't be > 100
        return hours_studied, exam_score

    
    def _generate_skewed_residuals_data(self, n_points=5000):
        """
        Generates data that produces a smooth, right-skewed residual distribution
        like the example image.
        """
        np.random.seed(123)
        # 1. Independent variable (e.g., Ad Spend)
        ad_spend = np.random.uniform(1000, 10000, n_points)
        
        # 2. Generate errors from a log-normal distribution to get the desired shape.
        #    sigma controls the amount of skew.
        skewed_errors = np.random.lognormal(mean=0, sigma=0.8, size=n_points)
        
        # 3. Scale and center the errors so they behave like proper residuals (mean=0)
        skewed_errors = skewed_errors * 8000  # Scale up the magnitude
        skewed_errors = skewed_errors - np.mean(skewed_errors) # Center on zero
        
        # 4. Create the dependent variable (e.g., Monthly Sales)
        monthly_sales = 10000 + 5 * ad_spend + skewed_errors
        
        return ad_spend, monthly_sales

    def _fit_linear_model(self, X, y, fit_partial=False):
        """Fit a linear regression model and return predictions and residuals"""
        X_reshaped = X.reshape(-1, 1)
        model = LinearRegression()
        
        if fit_partial:
            # Fit only to the first part of the data (first 40% of points)
            n_partial = int(len(X) * 0.4)
            model.fit(X_reshaped[:n_partial], y[:n_partial])
        else:
            model.fit(X_reshaped, y)
            
        y_pred = model.predict(X_reshaped)
        residuals = y - y_pred
        r2 = r2_score(y, y_pred)
        return y_pred, residuals, r2, model
    
    def visualize_linearity_assumption(self, show_residuals=True):
        """
        Create interactive visualization of linearity assumption.
        
        Parameters:
        -----------
        show_residuals : bool
            If True, creates subplot with residual plots
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        # Generate data
        pizzas, happiness_linear = self._generate_linear_data()
        volume, happiness_nonlinear = self._generate_nonlinear_data()
        
        # Fit linear models
        pred_linear, resid_linear, r2_linear, model_linear = self._fit_linear_model(pizzas, happiness_linear)
        pred_nonlinear, resid_nonlinear, r2_nonlinear, model_nonlinear = self._fit_linear_model(volume, happiness_nonlinear, fit_partial=True)
        
        if show_residuals:
            # Create subplots: 2x2 grid
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    '✅ Good Example: Pizza vs Happiness (Linear)',
                    '❌ Bad Example: Volume vs Happiness (Non-linear)',
                    'Residuals vs Predicted - Good Example',
                    'Residuals vs Predicted - Bad Example'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
        else:
            # Create subplots: 1x2 grid
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    '✅ Good Example: Pizza vs Happiness (Linear)',
                    '❌ Bad Example: Volume vs Happiness (Non-linear)'
                ],
                horizontal_spacing=0.1
            )
        
        # Good example - Linear relationship
        fig.add_trace(
            go.Scatter(
                x=pizzas, 
                y=happiness_linear,
                mode='markers',
                name='Data Points',
                marker=dict(color=self.colors['good'], size=6, opacity=0.7),
                hovertemplate='Pizza: %{x:.1f}<br>Happiness: %{y:.1f}<extra></extra>',
                legendgroup='good',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pizzas,
                y=pred_linear,
                mode='lines',
                name=f'Linear Fit (R² = {r2_linear:.3f})',
                line=dict(color=self.colors['good'], width=3),
                hovertemplate='Pizza: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>',
                legendgroup='good',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Bad example - Non-linear relationship
        fig.add_trace(
            go.Scatter(
                x=volume,
                y=happiness_nonlinear,
                mode='markers',
                name='Data Points',
                marker=dict(color=self.colors['bad'], size=6, opacity=0.7),
                hovertemplate='Volume: %{x:.1f}<br>Happiness: %{y:.1f}<extra></extra>',
                legendgroup='bad',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=volume,
                y=pred_nonlinear,
                mode='lines',
                name=f'Linear Fit (R² = {r2_nonlinear:.3f})',
                line=dict(color=self.colors['bad'], width=3),
                hovertemplate='Volume: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>',
                legendgroup='bad',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add annotation to show which part was used for fitting
        n_partial = int(len(volume) * 0.4)
        fig.add_trace(
            go.Scatter(
                x=volume[:n_partial],
                y=happiness_nonlinear[:n_partial],
                mode='markers',
                name='Fitting Points',
                marker=dict(color='orange', size=8, symbol='diamond', opacity=0.9),
                hovertemplate='Volume: %{x:.1f}<br>Happiness: %{y:.1f}<br><b>Used for fitting</b><extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        if show_residuals:
            # Residuals plot - Good example (should show random scatter around 0)
            fig.add_trace(
                go.Scatter(
                    x=pred_linear,  # Predicted values on x-axis
                    y=resid_linear, # Residuals on y-axis
                    mode='markers',
                    name='Residuals - Linear Data',
                    marker=dict(color=self.colors['good'], size=6, opacity=0.7),
                    hovertemplate='Predicted Value: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>',
                    legendgroup='resid_good',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add horizontal line at y=0 for good example
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.7, row=2, col=1)
            
            # Residuals plot - Bad example (should show curved pattern)
            fig.add_trace(
                go.Scatter(
                    x=pred_nonlinear,  # Predicted values on x-axis
                    y=resid_nonlinear, # Residuals on y-axis
                    mode='markers',
                    name='Residuals - Non-linear Data',
                    marker=dict(color=self.colors['bad'], size=6, opacity=0.7),
                    hovertemplate='Predicted Value: %{x:.1f}<br>Residual: %{y:.1f}<extra></extra>',
                    legendgroup='resid_bad',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Add horizontal line at y=0 for bad example
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.7, row=2, col=2)
        
        # Update layout
        height = 800 if show_residuals else 400
        fig.update_layout(
            title={
                'text': 'Regression Linearity Assumption Visualization',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=height,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor=self.colors['background']
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Number of Pizzas", row=1, col=1)
        fig.update_yaxes(title_text="Happiness Level", row=1, col=1)
        fig.update_xaxes(title_text="Party Volume Level", row=1, col=2)
        fig.update_yaxes(title_text="Happiness Level", row=1, col=2)
        
        if show_residuals:
            fig.update_xaxes(title_text="Predicted Values", row=2, col=1)
            fig.update_yaxes(title_text="Residuals", row=2, col=1)
            fig.update_xaxes(title_text="Predicted Values", row=2, col=2)
            fig.update_yaxes(title_text="Residuals", row=2, col=2)
        
        # Add annotations to explain the patterns
        annotations = [
            dict(
                x=0.25, y=1.0,
                xref="paper", yref="paper",
                text="<b>Random scatter around zero</b><br>✅ Linearity assumption satisfied",
                showarrow=False,
                bgcolor="rgba(46, 139, 87, 0.1)",
                bordercolor=self.colors['good'],
                borderwidth=1,
                font=dict(size=10)
            ) if show_residuals else None,
            dict(
                x=0.75, y=1.0,
                xref="paper", yref="paper", 
                text="<b>Clear curved pattern</b><br>❌ Linearity assumption violated",
                showarrow=False,
                bgcolor="rgba(220, 20, 60, 0.1)",
                bordercolor=self.colors['bad'],
                borderwidth=1,
                font=dict(size=10)
            ) if show_residuals else None,
            dict(
                x=0.75, y=0.85,
                xref="paper", yref="paper",
                text="<b>◆ Orange points:</b> Used for fitting<br><b>Line fits first part well</b><br>but fails on the rest",
                showarrow=False,
                bgcolor="rgba(255, 165, 0, 0.1)",
                bordercolor="orange",
                borderwidth=1,
                font=dict(size=9)
            )
        ]
        
        if show_residuals:
            fig.update_layout(annotations=[ann for ann in annotations if ann])
        else:
            # Only show the fitting annotation for the non-residuals view
            fig.update_layout(annotations=[annotations[2]])
        
        return fig
    
    def visualize_homoscedasticity_assumption(self, show_residuals=True):
        """
        Create interactive visualization of homoscedasticity assumption.
        
        Parameters:
        -----------
        show_residuals : bool
            If True, creates subplot with residual plots
            
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        # Generate data
        income_homo, food_homo = self._generate_homoscedastic_data()
        income_hetero, food_hetero = self._generate_heteroscedastic_data()
        
        # Fit linear models
        pred_homo, resid_homo, r2_homo, model_homo = self._fit_linear_model(income_homo, food_homo)
        pred_hetero, resid_hetero, r2_hetero, model_hetero = self._fit_linear_model(income_hetero, food_hetero)
        
        if show_residuals:
            # Create subplots: 2x2 grid
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    '✅ Good Example: Homoscedastic (Consistent Errors)',
                    '❌ Bad Example: Heteroscedastic (Funnel Pattern)',
                    'Residuals vs Predicted - Homoscedastic',
                    'Residuals vs Predicted - Heteroscedastic'
                ],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
        else:
            # Create subplots: 1x2 grid
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    '✅ Good Example: Homoscedastic (Consistent Errors)',
                    '❌ Bad Example: Heteroscedastic (Funnel Pattern)'
                ],
                horizontal_spacing=0.1
            )
        
        # Good example - Homoscedastic relationship
        fig.add_trace(
            go.Scatter(
                x=income_homo/1000,  # Convert to thousands for readability
                y=food_homo,
                mode='markers',
                name='Data Points',
                marker=dict(color=self.colors['good'], size=6, opacity=0.7),
                hovertemplate='Income: $%{x:.0f}k<br>Food Spending: $%{y:.0f}<extra></extra>',
                legendgroup='good',
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=income_homo/1000,
                y=pred_homo,
                mode='lines',
                name=f'Linear Fit (R² = {r2_homo:.3f})',
                line=dict(color=self.colors['good'], width=3),
                hovertemplate='Income: $%{x:.0f}k<br>Predicted: $%{y:.0f}<extra></extra>',
                legendgroup='good',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Bad example - Heteroscedastic relationship
        fig.add_trace(
            go.Scatter(
                x=income_hetero/1000,
                y=food_hetero,
                mode='markers',
                name='Data Points',
                marker=dict(color=self.colors['bad'], size=6, opacity=0.7),
                hovertemplate='Income: $%{x:.0f}k<br>Food Spending: $%{y:.0f}<extra></extra>',
                legendgroup='bad',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=income_hetero/1000,
                y=pred_hetero,
                mode='lines',
                name=f'Linear Fit (R² = {r2_hetero:.3f})',
                line=dict(color=self.colors['bad'], width=3),
                hovertemplate='Income: $%{x:.0f}k<br>Predicted: $%{y:.0f}<extra></extra>',
                legendgroup='bad',
                showlegend=False
            ),
            row=1, col=2
        )
        
        if show_residuals:
            # Residuals plot - Good example (should show consistent spread around 0)
            fig.add_trace(
                go.Scatter(
                    x=pred_homo,
                    y=resid_homo,
                    mode='markers',
                    name='Residuals - Homoscedastic',
                    marker=dict(color=self.colors['good'], size=6, opacity=0.7),
                    hovertemplate='Predicted: $%{x:.0f}<br>Residual: $%{y:.0f}<extra></extra>',
                    legendgroup='resid_good',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add horizontal line at y=0 for good example
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.7, row=2, col=1)
            
            # Residuals plot - Bad example (should show funnel/cone pattern)
            fig.add_trace(
                go.Scatter(
                    x=pred_hetero,
                    y=resid_hetero,
                    mode='markers',
                    name='Residuals - Heteroscedastic',
                    marker=dict(color=self.colors['bad'], size=6, opacity=0.7),
                    hovertemplate='Predicted: $%{x:.0f}<br>Residual: $%{y:.0f}<extra></extra>',
                    legendgroup='resid_bad',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Add horizontal line at y=0 for bad example
            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                         opacity=0.7, row=2, col=2)
        
        # Update layout
        height = 800 if show_residuals else 400
        fig.update_layout(
            title={
                'text': 'Regression Homoscedasticity Assumption Visualization',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=height,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor=self.colors['background']
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Annual Income ($k)", row=1, col=1)
        fig.update_yaxes(title_text="Annual Food Spending ($)", row=1, col=1)
        fig.update_xaxes(title_text="Annual Income ($k)", row=1, col=2)
        fig.update_yaxes(title_text="Annual Food Spending ($)", row=1, col=2)
        
        if show_residuals:
            fig.update_xaxes(title_text="Predicted Values ($)", row=2, col=1)
            fig.update_yaxes(title_text="Residuals ($)", row=2, col=1)
            fig.update_xaxes(title_text="Predicted Values ($)", row=2, col=2)
            fig.update_yaxes(title_text="Residuals ($)", row=2, col=2)
        
        # Add annotations to explain the patterns
        annotations = [
            dict(
                x=0.15, y=0.01,
                xref="paper", yref="paper",
                text="<b>Consistent spread at all levels</b><br>✅ Homoscedasticity satisfied",
                showarrow=False,
                bgcolor="rgba(46, 139, 87, 0.1)",
                bordercolor=self.colors['good'],
                borderwidth=1,
                font=dict(size=10)
            ) if show_residuals else dict(
                x=0.25, y=0.48,
                xref="paper", yref="paper",
                text="<b>Consistent spread at all income levels</b><br>✅ Homoscedasticity satisfied",
                showarrow=False,
                bgcolor="rgba(46, 139, 87, 0.1)",
                bordercolor=self.colors['good'],
                borderwidth=1,
                font=dict(size=10)
            ),
            dict(
                x=0.70, y=0.01,
                xref="paper", yref="paper", 
                text="<b>Funnel/Cone pattern</b><br>❌ Heteroscedasticity detected",
                showarrow=False,
                bgcolor="rgba(220, 20, 60, 0.1)",
                bordercolor=self.colors['bad'],
                borderwidth=1,
                font=dict(size=10)
            ) if show_residuals else dict(
                x=0.75, y=0.48,
                xref="paper", yref="paper", 
                text="<b>Funnel/Cone pattern</b><br>❌ Heteroscedasticity detected",
                showarrow=False,
                bgcolor="rgba(220, 20, 60, 0.1)",
                bordercolor=self.colors['bad'],
                borderwidth=1,
                font=dict(size=10)
            )
        ]
        
        fig.update_layout(annotations=annotations)
        
        return fig

    def visualize_autocorrelation_assumption(self):
        """
        Create interactive visualization of the independence of errors assumption (autocorrelation).
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        # Generate data
        time_ind, pred_ind, target_ind = self._generate_independent_errors_data()
        time_auto, pred_auto, target_auto = self._generate_autocorrelated_errors_data()
        
        # Fit models and get residuals
        _, resid_ind, _, _ = self._fit_linear_model(pred_ind, target_ind)
        _, resid_auto, _, _ = self._fit_linear_model(pred_auto, target_auto)
        
        # Create subplots: 1x2 grid
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                '✅ Good Example: Independent Errors',
                '❌ Bad Example: Autocorrelated Errors (Cyclical Pattern)'
            ],
            horizontal_spacing=0.1
        )
        
        # Good example - Independent errors (random scatter)
        fig.add_trace(
            go.Scatter(
                x=time_ind,
                y=resid_ind,
                mode='markers',
                name='Residuals - Independent',
                marker=dict(color=self.colors['good'], size=6, opacity=0.7),
                hovertemplate='Time: %{x}<br>Residual: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add connecting lines to emphasize order, but make them subtle
        fig.add_trace(
            go.Scatter(
                x=time_ind, y=resid_ind, mode='lines',
                line=dict(color=self.colors['good'], width=1, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add horizontal line at y=0 for good example
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     opacity=0.7, row=1, col=1)
        
        # Bad example - Autocorrelated errors (cyclical pattern)
        fig.add_trace(
            go.Scatter(
                x=time_auto,
                y=resid_auto,
                mode='markers',
                name='Residuals - Autocorrelated',
                marker=dict(color=self.colors['bad'], size=6, opacity=0.7),
                hovertemplate='Time: %{x}<br>Residual: %{y:.2f}<extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add connecting lines to emphasize the cyclical pattern
        fig.add_trace(
            go.Scatter(
                x=time_auto, y=resid_auto, mode='lines',
                line=dict(color=self.colors['bad'], width=1),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add horizontal line at y=0 for bad example
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     opacity=0.7, row=1, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Regression Assumption: Independence of Errors',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor=self.colors['background']
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time / Order of Observation", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Time / Order of Observation", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        # Add annotations
        annotations = [
            dict(
                x=0.25, y=0.98,
                xref="paper", yref="paper",
                text="<b>Random scatter around zero</b><br>✅ No pattern, assumption holds",
                showarrow=False,
                bgcolor="rgba(46, 139, 87, 0.1)",
                bordercolor=self.colors['good'],
                borderwidth=1,
                font=dict(size=12)
            ),
            dict(
                x=0.75, y=0.98,
                xref="paper", yref="paper", 
                text="<b>Clear down-up-down pattern</b><br>❌ Autocorrelation detected",
                showarrow=False,
                bgcolor="rgba(220, 20, 60, 0.1)",
                bordercolor=self.colors['bad'],
                borderwidth=1,
                font=dict(size=12)
            )
        ]
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def visualize_normality_assumption(self):
        """
        Create interactive visualization for the normality of residuals assumption.
        Uses histograms and Q-Q plots for diagnosis.
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        
        # Generate data based on the provided examples
        hours_studied, exam_scores = self._generate_normal_residuals_data()
        ad_spend, monthly_sales = self._generate_skewed_residuals_data()
        
        # Fit models and get residuals
        _, resid_norm, _, _ = self._fit_linear_model(hours_studied, exam_scores)
        _, resid_skew, _, _ = self._fit_linear_model(ad_spend, monthly_sales)
        
        # Create subplots: 2 rows (Histogram, Q-Q Plot), 2 columns (Good, Bad)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '✅ Good: Exam Scores vs. Hours Studied',
                '❌ Bad: Monthly Sales vs. Ad Spend',
                'Q-Q Plot - Exam Scores',
                'Q-Q Plot - Monthly Sales'
            ],
            vertical_spacing=0.18,
            horizontal_spacing=0.1
        )
        
        # --- ROW 1: HISTOGRAMS ---
        
        # Good example - Histogram (should be bell-shaped)
        fig.add_trace(
            go.Histogram(
                x=resid_norm,
                name='Normal Residuals',
                marker_color=self.colors['good'],
                opacity=0.7,
                showlegend=False,
                histnorm='probability density'
            ),
            row=1, col=1
        )
        
        # Bad example - Histogram (should be skewed)
        fig.add_trace(
            go.Histogram(
                x=resid_skew,
                name='Skewed Residuals',
                marker_color=self.colors['bad'],
                opacity=0.7,
                showlegend=False,
                histnorm='probability density',
                nbinsx=100
            ),
            row=1, col=2
        )

        # Overlay normal distribution curves
        x_norm_curve = np.linspace(resid_norm.min(), resid_norm.max(), 100)
        y_norm_curve = stats.norm.pdf(x_norm_curve, np.mean(resid_norm), np.std(resid_norm))
        fig.add_trace(go.Scatter(x=x_norm_curve, y=y_norm_curve, mode='lines', line=dict(color='black', width=2), name='Normal Curve'), row=1, col=1)

        x_skew_curve = np.linspace(resid_skew.min(), resid_skew.max(), 100)
        y_skew_curve = stats.norm.pdf(x_skew_curve, np.mean(resid_skew), np.std(resid_skew))
        fig.add_trace(go.Scatter(x=x_skew_curve, y=y_skew_curve, mode='lines', line=dict(color='black', width=2, dash='dash'), name='Normal Curve'), row=1, col=2)
        
        # --- ROW 2: Q-Q PLOTS ---
        
        # Good example - Q-Q Plot (should be a straight line)
        qq_norm = stats.probplot(resid_norm, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_norm[0][0], y=qq_norm[0][1],
                mode='markers',
                marker=dict(color=self.colors['good']),
                showlegend=False,
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=qq_norm[0][0], y=qq_norm[1][0] * qq_norm[0][0] + qq_norm[1][1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Bad example - Q-Q Plot (should deviate from the line)
        qq_skew = stats.probplot(resid_skew, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_skew[0][0], y=qq_skew[0][1],
                mode='markers',
                marker=dict(color=self.colors['bad']),
                showlegend=False,
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=qq_skew[0][0], y=qq_skew[1][0] * qq_skew[0][0] + qq_skew[1][1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # --- LAYOUT AND ANNOTATIONS ---
        
        fig.update_layout(
            title={
                'text': 'Regression Normality of Residuals Assumption',
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}
            },
            height=800,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor=self.colors['background']
        )
        
        # Update axes
        fig.update_xaxes(title_text="Residual (Predicted - Actual Score)", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_xaxes(title_text="Residual (Predicted - Actual Sales)", row=1, col=2)
        fig.update_yaxes(title_text="Density", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles (Normal)", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles (Residuals)", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles (Normal)", row=2, col=2)
        fig.update_yaxes(title_text="Sample Quantiles (Residuals)", row=2, col=2)

        # Add annotations
        annotations = [
            dict(x=0.25, y=1.0, 
                 xref="paper", 
                 yref="paper", 
                 text="<b>Symmetrical Bell Shape</b><br>✅ Normality holds", 
                 showarrow=False, 
                 bgcolor="rgba(46, 139, 87, 0.1)",
                 font=dict(size=11)),
            dict(x=0.75, y=1.0, 
                 xref="paper", 
                 yref="paper", 
                 text="<b>Right-Skewed by Sales Spikes</b><br>❌ Normality violated", 
                 showarrow=False, 
                 bgcolor="rgba(220, 20, 60, 0.1)",
                 font=dict(size=11)),
            dict(x=0.25, y=0.40, 
                 xref="paper", 
                 yref="paper", 
                 text="<b>Points follow the line closely</b><br>✅ Confirms normality", 
                 showarrow=False, 
                 bgcolor="rgba(46, 139, 87, 0.1)", 
                 font=dict(size=11)),
            dict(x=0.75, y=0.40, 
                 xref="paper", 
                 yref="paper", 
                 text="<b>Points curve away at the right tail</b><br>❌ Confirms non-normality", 
                 showarrow=False, 
                 bgcolor="rgba(220, 20, 60, 0.1)", 
                 font=dict(size=11)),
        ]
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def create_diagnostic_summary(self):
        """Create a summary of diagnostic information"""
        pizzas, happiness_linear = self._generate_linear_data()
        volume, happiness_nonlinear = self._generate_nonlinear_data()
        
        _, _, r2_linear, _ = self._fit_linear_model(pizzas, happiness_linear)
        _, _, r2_nonlinear, _ = self._fit_linear_model(volume, happiness_nonlinear, fit_partial=True)
        
        summary = {
            'Linear Example (Pizza)': {
                'R²': f'{r2_linear:.3f}',
                'Pattern': 'Random residuals',
                'Interpretation': 'Good linear fit'
            },
            'Non-linear Example (Volume)': {
                'R²': f'{r2_nonlinear:.3f}',
                'Pattern': 'Curved residuals',
                'Interpretation': 'Linear model inadequate'
            }
        }
        
        return summary

def visualize_linearity_assumption(show_residuals=True):
    """
    Convenience function to create linearity assumption visualization.
    
    Parameters:
    -----------
    show_residuals : bool, default=True
        Whether to show residual plots
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot showing good vs bad examples of linearity
        
    Example:
    --------
    >>> fig = visualize_linearity_assumption()
    >>> fig.show()
    """
    visualizer = RegressionAssumptionVisualizer()
    return visualizer.visualize_linearity_assumption(show_residuals=show_residuals)

def visualize_homoscedasticity_assumption(show_residuals=True):
    """
    Convenience function to create homoscedasticity assumption visualization.
    
    Parameters:
    -----------
    show_residuals : bool, default=True
        Whether to show residual plots
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot showing good vs bad examples of homoscedasticity
        
    Example:
    --------
    >>> fig = visualize_homoscedasticity_assumption()
    >>> fig.show()
    """
    visualizer = RegressionAssumptionVisualizer()
    return visualizer.visualize_homoscedasticity_assumption(show_residuals=show_residuals)

def visualize_autocorrelation_assumption():
    """
    Convenience function to create autocorrelation assumption visualization.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot showing good vs bad examples of autocorrelation
        
    Example:
    --------
    >>> fig = visualize_autocorrelation_assumption()
    >>> fig.show()
    """
    visualizer = RegressionAssumptionVisualizer()
    return visualizer.visualize_autocorrelation_assumption()

def visualize_normality_assumption():
    """
    Convenience function to create normality of residuals assumption visualization.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plot showing good vs bad examples of normality
        
    Example:
    --------
    >>> fig = visualize_normality_assumption()
    >>> fig.show()
    """
    visualizer = RegressionAssumptionVisualizer()
    return visualizer.visualize_normality_assumption()


# Example usage
if __name__ == "__main__":
    # Create the linearity visualization
    print("Creating linearity assumption visualization...")
    fig_linearity = visualize_linearity_assumption(show_residuals=True)
    #fig_linearity.show()
    
    # Create the homoscedasticity visualization
    print("Creating homoscedasticity assumption visualization...")
    fig_homoscedasticity = visualize_homoscedasticity_assumption(show_residuals=True)
    #fig_homoscedasticity.show()
    
    # Create the autocorrelation visualization
    print("Creating autocorrelation assumption visualization...")
    fig_autocorrelation = visualize_autocorrelation_assumption()
    #fig_autocorrelation.show()

    fig_normality = visualize_normality_assumption()
    #fig_normality.show()