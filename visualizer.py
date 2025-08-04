import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
from scipy import stats

class DataVisualizer:
    """
    Handles automatic visualization generation for data analysis.
    """
    
    def __init__(self):
        """Initialize the DataVisualizer with default styling."""
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
    
    def generate_auto_visualizations(self, df: pd.DataFrame) -> Dict[str, Dict[str, go.Figure]]:
        """
        Generate comprehensive automatic visualizations based on data types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of visualization categories and figures
        """
        visualizations = {
            'distributions': {},
            'relationships': {},
            'categorical': {},
            'outliers': {},
            'missing_values': {}
        }
        
        # Get column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Generate distribution plots
        visualizations['distributions'] = self._create_distribution_plots(df, numerical_cols)
        
        # Generate relationship plots
        visualizations['relationships'] = self._create_relationship_plots(df, numerical_cols)
        
        # Generate categorical plots
        visualizations['categorical'] = self._create_categorical_plots(df, categorical_cols, numerical_cols)
        
        # Generate outlier plots
        visualizations['outliers'] = self._create_outlier_plots(df, numerical_cols)
        
        # Generate missing value plots
        visualizations['missing_values'] = self._create_missing_value_plots(df)
        
        # Generate time series plots if datetime columns exist
        if datetime_cols:
            visualizations['time_series'] = self._create_time_series_plots(df, datetime_cols, numerical_cols)
        
        return visualizations
    
    def _create_distribution_plots(self, df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create distribution plots for numerical columns."""
        plots = {}
        
        for col in numerical_cols[:8]:  # Limit to first 8 columns to avoid overload
            if df[col].notna().sum() == 0:
                continue
            
            # Histogram with KDE
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=df[col].dropna(),
                nbinsx=30,
                name='Histogram',
                opacity=0.7,
                yaxis='y',
                histnorm='probability density'
            ))
            
            # Add KDE if possible
            try:
                from scipy.stats import gaussian_kde
                data = df[col].dropna()
                if len(data) > 1:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    kde_values = kde(x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        name='KDE',
                        line=dict(color='red', width=2),
                        yaxis='y2'
                    ))
            except:
                pass
            
            fig.update_layout(
                title=f'Distribution of {col}',
                xaxis_title=col,
                yaxis_title='Frequency',
                template=self.template,
                yaxis2=dict(overlaying='y', side='right', title='Density'),
                showlegend=True
            )
            
            plots[f'{col}_distribution'] = fig
        
        return plots
    
    def _create_relationship_plots(self, df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create relationship plots between numerical variables."""
        plots = {}
        
        if len(numerical_cols) < 2:
            return plots
        
        # Correlation heatmap
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            plots['correlation_heatmap'] = self.create_correlation_heatmap(corr_matrix)
        
        # Scatter plot matrix (for small number of variables)
        if 2 <= len(numerical_cols) <= 6:
            plots['scatter_matrix'] = self._create_scatter_matrix(df, numerical_cols)
        
        # Top correlations scatter plots
        if len(numerical_cols) >= 2:
            top_correlations = self._get_top_correlations(df[numerical_cols], n=3)
            for i, (col1, col2, corr_val) in enumerate(top_correlations):
                fig = px.scatter(
                    df, x=col1, y=col2,
                    title=f'{col1} vs {col2} (Correlation: {corr_val:.3f})',
                    template=self.template,
                    trendline="ols"
                )
                plots[f'scatter_{col1}_vs_{col2}'] = fig
        
        return plots
    
    def _create_categorical_plots(self, df: pd.DataFrame, categorical_cols: List[str], 
                                numerical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create plots for categorical variables."""
        plots = {}
        
        # Bar charts for categorical variables
        for col in categorical_cols[:5]:  # Limit to first 5
            if df[col].notna().sum() == 0:
                continue
            
            value_counts = df[col].value_counts().head(20)  # Top 20 categories
            
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f'Distribution of {col}',
                labels={'x': 'Count', 'y': col},
                template=self.template
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, len(value_counts) * 25)
            )
            
            plots[f'{col}_bar_chart'] = fig
        
        # Box plots: categorical vs numerical
        for cat_col in categorical_cols[:3]:
            for num_col in numerical_cols[:3]:
                if df[cat_col].notna().sum() == 0 or df[num_col].notna().sum() == 0:
                    continue
                
                # Limit categories to prevent overcrowding
                top_categories = df[cat_col].value_counts().head(10).index
                filtered_df = df[df[cat_col].isin(top_categories)]
                
                fig = px.box(
                    filtered_df, x=cat_col, y=num_col,
                    title=f'{num_col} by {cat_col}',
                    template=self.template
                )
                
                fig.update_xaxes(tickangle=45)
                plots[f'box_{num_col}_by_{cat_col}'] = fig
        
        return plots
    
    def _create_outlier_plots(self, df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create plots specifically for outlier detection."""
        plots = {}
        
        for col in numerical_cols[:6]:  # Limit to first 6 columns
            if df[col].notna().sum() == 0:
                continue
            
            data = df[col].dropna()
            
            # Box plot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=data,
                name=col,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            fig_box.update_layout(
                title=f'Outliers in {col}',
                yaxis_title=col,
                template=self.template
            )
            
            plots[f'{col}_outliers'] = fig_box
        
        return plots
    
    def _create_missing_value_plots(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create visualizations for missing value patterns."""
        plots = {}
        
        # Missing value bar chart
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        
        if len(missing_counts) > 0:
            fig = px.bar(
                x=missing_counts.values,
                y=missing_counts.index,
                orientation='h',
                title='Missing Values by Column',
                labels={'x': 'Missing Count', 'y': 'Column'},
                template=self.template
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            plots['missing_values_bar'] = fig
        
        # Missing value heatmap
        if df.isnull().sum().sum() > 0:
            missing_matrix = df.isnull().astype(int)
            
            fig = go.Figure(data=go.Heatmap(
                z=missing_matrix.values.T,
                x=list(range(len(df))),
                y=missing_matrix.columns,
                colorscale=[[0, 'lightblue'], [1, 'darkred']],
                showscale=True,
                colorbar=dict(title="Missing")
            ))
            
            fig.update_layout(
                title='Missing Value Pattern',
                xaxis_title='Row Index',
                yaxis_title='Columns',
                template=self.template
            )
            
            plots['missing_pattern_heatmap'] = fig
        
        return plots
    
    def _create_time_series_plots(self, df: pd.DataFrame, datetime_cols: List[str], 
                                numerical_cols: List[str]) -> Dict[str, go.Figure]:
        """Create time series plots."""
        plots = {}
        
        for date_col in datetime_cols:
            for num_col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                if df[date_col].notna().sum() == 0 or df[num_col].notna().sum() == 0:
                    continue
                
                # Sort by date
                df_sorted = df.sort_values(date_col)
                
                fig = px.line(
                    df_sorted, x=date_col, y=num_col,
                    title=f'{num_col} over time',
                    template=self.template
                )
                
                plots[f'{num_col}_timeseries'] = fig
        
        return plots
    
    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame, 
                                 title: str = "Correlation Matrix") -> go.Figure:
        """Create an interactive correlation heatmap."""
        
        # Create annotations for correlation values
        annotations = []
        for i, row in enumerate(corr_matrix.index):
            for j, col in enumerate(corr_matrix.columns):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=str(round(corr_matrix.loc[row, col], 2)),
                        showarrow=False,
                        font=dict(color="white" if abs(corr_matrix.loc[row, col]) > 0.5 else "black")
                    )
                )
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template=self.template,
            width=600,
            height=600
        )
        
        return fig
    
    def _create_scatter_matrix(self, df: pd.DataFrame, numerical_cols: List[str]) -> go.Figure:
        """Create a scatter plot matrix."""
        
        # Use only a subset if too many columns
        cols_to_use = numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols
        
        fig = go.Figure()
        
        # Create subplots
        n_cols = len(cols_to_use)
        subplot_titles = []
        
        for i in range(n_cols):
            for j in range(n_cols):
                if i == j:
                    subplot_titles.append(f'{cols_to_use[i]} Distribution')
                else:
                    subplot_titles.append(f'{cols_to_use[j]} vs {cols_to_use[i]}')
        
        fig = make_subplots(
            rows=n_cols, cols=n_cols,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            shared_yaxes=True
        )
        
        for i, col_y in enumerate(cols_to_use):
            for j, col_x in enumerate(cols_to_use):
                if i == j:
                    # Diagonal: histogram
                    fig.add_trace(
                        go.Histogram(x=df[col_x].dropna(), showlegend=False),
                        row=i+1, col=j+1
                    )
                else:
                    # Off-diagonal: scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=df[col_x], y=df[col_y],
                            mode='markers',
                            showlegend=False,
                            opacity=0.6,
                            marker=dict(size=3)
                        ),
                        row=i+1, col=j+1
                    )
        
        fig.update_layout(
            title="Scatter Plot Matrix",
            template=self.template,
            height=150 * n_cols,
            width=150 * n_cols
        )
        
        return fig
    
    def _get_top_correlations(self, df: pd.DataFrame, n: int = 5) -> List[tuple]:
        """Get top n correlations from correlation matrix."""
        corr_matrix = df.corr()
        
        # Get upper triangle of correlation matrix
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find top correlations
        correlations = []
        for col in upper_triangle.columns:
            for row in upper_triangle.index:
                if pd.notna(upper_triangle.loc[row, col]):
                    correlations.append((row, col, abs(upper_triangle.loc[row, col])))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        return correlations[:n]
    
    def create_custom_plot(self, df: pd.DataFrame, plot_type: str, 
                          x_col: str, y_col: str = None, **kwargs) -> go.Figure:
        """
        Create a custom plot based on user specifications.
        
        Args:
            df: Input DataFrame
            plot_type: Type of plot ('scatter', 'line', 'bar', 'box', 'histogram')
            x_col: X-axis column
            y_col: Y-axis column (optional for some plot types)
            **kwargs: Additional plot parameters
            
        Returns:
            Plotly figure
        """
        
        if plot_type == 'scatter':
            fig = px.scatter(df, x=x_col, y=y_col, template=self.template, **kwargs)
        elif plot_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, template=self.template, **kwargs)
        elif plot_type == 'bar':
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, template=self.template, **kwargs)
            else:
                value_counts = df[x_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           template=self.template, **kwargs)
        elif plot_type == 'box':
            fig = px.box(df, x=x_col, y=y_col, template=self.template, **kwargs)
        elif plot_type == 'histogram':
            fig = px.histogram(df, x=x_col, template=self.template, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        return fig