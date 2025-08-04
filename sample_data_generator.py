"""
Sample Data Generator for Testing the Data Analytics Application

This module creates various types of synthetic datasets for testing
different features and capabilities of the analytics tool.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional
import json

class SampleDataGenerator:
    """Generate various types of sample datasets for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_sales_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """
        Generate a realistic sales dataset.
        
        Args:
            n_rows: Number of rows to generate
            
        Returns:
            DataFrame with sales data
        """
        
        # Date range
        start_date = datetime.now() - timedelta(days=365)
        dates = [start_date + timedelta(days=x) for x in range(n_rows)]
        
        # Product categories and names
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports']
        products = {
            'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera'],
            'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'],
            'Home & Garden': ['Chair', 'Table', 'Lamp', 'Plant', 'Decoration'],
            'Books': ['Fiction', 'Non-Fiction', 'Textbook', 'Biography', 'Self-Help'],
            'Sports': ['Basketball', 'Tennis Racket', 'Running Shoes', 'Yoga Mat', 'Weights']
        }
        
        # Regions and sales reps
        regions = ['North', 'South', 'East', 'West', 'Central']
        sales_reps = [f'Rep_{i:03d}' for i in range(1, 21)]
        
        # Generate data
        data = []
        for i in range(n_rows):
            category = np.random.choice(categories)
            product = np.random.choice(products[category])
            region = np.random.choice(regions)
            rep = np.random.choice(sales_reps)
            
            # Price varies by category
            price_ranges = {
                'Electronics': (100, 2000),
                'Clothing': (20, 200),
                'Home & Garden': (50, 500),
                'Books': (10, 50),
                'Sports': (25, 300)
            }
            
            base_price = np.random.uniform(*price_ranges[category])
            quantity = np.random.randint(1, 10)
            
            # Add some seasonality and trends
            month = dates[i].month
            if category == 'Electronics' and month in [11, 12]:  # Holiday season
                base_price *= 1.2
                quantity = int(quantity * 1.5)
            elif category == 'Sports' and month in [4, 5, 6]:  # Spring/Summer
                base_price *= 1.1
                quantity = int(quantity * 1.3)
            
            revenue = base_price * quantity
            
            # Add some missing values occasionally
            customer_age = np.random.randint(18, 70) if random.random() > 0.05 else None
            customer_satisfaction = np.random.uniform(1, 5) if random.random() > 0.03 else None
            
            data.append({
                'Date': dates[i],
                'Category': category,
                'Product': product,
                'Region': region,
                'Sales_Rep': rep,
                'Quantity': quantity,
                'Unit_Price': round(base_price, 2),
                'Revenue': round(revenue, 2),
                'Customer_Age': customer_age,
                'Customer_Satisfaction': round(customer_satisfaction, 1) if customer_satisfaction else None,
                'Discount_Applied': np.random.choice([True, False], p=[0.3, 0.7]),
                'Return_Customer': np.random.choice([True, False], p=[0.6, 0.4])
            })
        
        df = pd.DataFrame(data)
        
        # Add some outliers
        outlier_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
        df.loc[outlier_indices, 'Revenue'] *= np.random.uniform(3, 10, size=len(outlier_indices))
        
        return df
    
    def generate_customer_data(self, n_rows: int = 500) -> pd.DataFrame:
        """
        Generate customer demographics and behavior data.
        
        Args:
            n_rows: Number of customers to generate
            
        Returns:
            DataFrame with customer data
        """
        
        # Customer segments
        segments = ['Premium', 'Standard', 'Budget', 'Enterprise']
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
                 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        
        data = []
        for i in range(n_rows):
            customer_id = f'CUST_{i+1:05d}'
            
            # Age influences other characteristics
            age = np.random.normal(40, 15)
            age = max(18, min(80, int(age)))
            
            # Income correlation with age (somewhat)
            base_income = 30000 + (age - 18) * 1000 + np.random.normal(0, 20000)
            annual_income = max(20000, int(base_income))
            
            # Segment based on income
            if annual_income > 100000:
                segment = np.random.choice(['Premium', 'Enterprise'], p=[0.7, 0.3])
            elif annual_income > 60000:
                segment = np.random.choice(['Premium', 'Standard'], p=[0.3, 0.7])
            else:
                segment = np.random.choice(['Standard', 'Budget'], p=[0.4, 0.6])
            
            # Purchase behavior
            if segment == 'Premium':
                avg_order_value = np.random.normal(200, 50)
                monthly_orders = np.random.poisson(8)
            elif segment == 'Enterprise':
                avg_order_value = np.random.normal(500, 100)
                monthly_orders = np.random.poisson(12)
            elif segment == 'Standard':
                avg_order_value = np.random.normal(100, 30)
                monthly_orders = np.random.poisson(4)
            else:  # Budget
                avg_order_value = np.random.normal(50, 20)
                monthly_orders = np.random.poisson(2)
            
            avg_order_value = max(10, avg_order_value)
            monthly_orders = max(0, monthly_orders)
            
            # Customer lifetime value
            months_active = np.random.randint(1, 36)
            clv = avg_order_value * monthly_orders * months_active
            
            # Add some missing values
            education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                       p=[0.3, 0.4, 0.2, 0.1]) if random.random() > 0.08 else None
            
            data.append({
                'Customer_ID': customer_id,
                'Age': age,
                'Annual_Income': annual_income,
                'City': np.random.choice(cities),
                'Segment': segment,
                'Education': education,
                'Months_Active': months_active,
                'Monthly_Orders': monthly_orders,
                'Avg_Order_Value': round(avg_order_value, 2),
                'Customer_Lifetime_Value': round(clv, 2),
                'Email_Subscriber': np.random.choice([True, False], p=[0.7, 0.3]),
                'Mobile_App_User': np.random.choice([True, False], p=[0.6, 0.4]),
                'Support_Tickets': np.random.poisson(2),
                'NPS_Score': np.random.randint(0, 11) if random.random() > 0.1 else None
            })
        
        return pd.DataFrame(data)
    
    def generate_marketing_data(self, n_rows: int = 2000) -> pd.DataFrame:
        """
        Generate marketing campaign performance data.
        
        Args:
            n_rows: Number of campaign records to generate
            
        Returns:
            DataFrame with marketing data
        """
        
        channels = ['Email', 'Social Media', 'Google Ads', 'Facebook Ads', 'Display', 'SEO']
        campaign_types = ['Acquisition', 'Retention', 'Upsell', 'Brand Awareness']
        
        data = []
        for i in range(n_rows):
            channel = np.random.choice(channels)
            campaign_type = np.random.choice(campaign_types)
            
            # Campaign duration
            start_date = datetime.now() - timedelta(days=np.random.randint(1, 180))
            duration = np.random.randint(7, 60)
            end_date = start_date + timedelta(days=duration)
            
            # Budget varies by channel
            budget_ranges = {
                'Email': (500, 5000),
                'Social Media': (1000, 10000),
                'Google Ads': (2000, 20000),
                'Facebook Ads': (1500, 15000),
                'Display': (3000, 25000),
                'SEO': (2000, 12000)
            }
            
            budget = np.random.uniform(*budget_ranges[channel])
            
            # Performance metrics vary by channel
            if channel == 'Email':
                impressions = np.random.randint(10000, 100000)
                click_rate = np.random.uniform(0.02, 0.08)
                conversion_rate = np.random.uniform(0.01, 0.05)
            elif channel in ['Google Ads', 'Facebook Ads']:
                impressions = np.random.randint(50000, 500000)
                click_rate = np.random.uniform(0.01, 0.05)
                conversion_rate = np.random.uniform(0.005, 0.03)
            else:
                impressions = np.random.randint(20000, 200000)
                click_rate = np.random.uniform(0.005, 0.03)
                conversion_rate = np.random.uniform(0.003, 0.02)
            
            clicks = int(impressions * click_rate)
            conversions = int(clicks * conversion_rate)
            
            # Revenue varies by campaign type
            if campaign_type == 'Acquisition':
                avg_revenue_per_conversion = np.random.uniform(50, 200)
            elif campaign_type == 'Upsell':
                avg_revenue_per_conversion = np.random.uniform(100, 500)
            elif campaign_type == 'Retention':
                avg_revenue_per_conversion = np.random.uniform(30, 150)
            else:  # Brand Awareness
                avg_revenue_per_conversion = np.random.uniform(20, 100)
            
            revenue = conversions * avg_revenue_per_conversion
            roi = (revenue - budget) / budget if budget > 0 else 0
            
            data.append({
                'Campaign_ID': f'CAMP_{i+1:05d}',
                'Campaign_Name': f'{channel}_{campaign_type}_{i+1}',
                'Channel': channel,
                'Campaign_Type': campaign_type,
                'Start_Date': start_date,
                'End_Date': end_date,
                'Duration_Days': duration,
                'Budget': round(budget, 2),
                'Impressions': impressions,
                'Clicks': clicks,
                'Conversions': conversions,
                'Click_Rate': round(click_rate, 4),
                'Conversion_Rate': round(conversion_rate, 4),
                'Cost_Per_Click': round(budget / clicks if clicks > 0 else 0, 2),
                'Cost_Per_Conversion': round(budget / conversions if conversions > 0 else 0, 2),
                'Revenue': round(revenue, 2),
                'ROI': round(roi, 3),
                'Quality_Score': np.random.uniform(1, 10) if random.random() > 0.05 else None
            })
        
        return pd.DataFrame(data)
    
    def generate_iot_sensor_data(self, n_rows: int = 5000) -> pd.DataFrame:
        """
        Generate IoT sensor data with time series characteristics.
        
        Args:
            n_rows: Number of sensor readings to generate
            
        Returns:
            DataFrame with IoT sensor data
        """
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(hours=n_rows//10)  # 10 readings per hour
        timestamps = [start_time + timedelta(minutes=6*i) for i in range(n_rows)]
        
        # Sensor locations
        locations = ['Building_A', 'Building_B', 'Building_C', 'Warehouse', 'Parking_Lot']
        sensor_types = ['Temperature', 'Humidity', 'Air_Quality', 'Motion', 'Light']
        
        data = []
        for i, timestamp in enumerate(timestamps):
            location = np.random.choice(locations)
            
            # Add daily patterns (higher activity during business hours)
            hour = timestamp.hour
            is_business_hours = 8 <= hour <= 18
            weekday = timestamp.weekday() < 5
            
            # Temperature (with daily cycle)
            base_temp = 20 + 5 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
            temperature = base_temp + np.random.normal(0, 2)
            
            # Humidity (inversely related to temperature)
            humidity = max(0, min(100, 60 - (temperature - 20) * 2 + np.random.normal(0, 5)))
            
            # Air quality (worse during business hours)
            base_air_quality = 150 if is_business_hours and weekday else 100
            air_quality_index = max(0, base_air_quality + np.random.normal(0, 30))
            
            # Motion detection (higher during business hours)
            motion_prob = 0.7 if is_business_hours and weekday else 0.1
            motion_detected = np.random.random() < motion_prob
            
            # Light level (correlated with time of day and motion)
            if 6 <= hour <= 20:  # Daylight hours
                base_light = 800 + 200 * np.sin(np.pi * (hour - 6) / 14)
            else:
                base_light = 50
            
            if motion_detected:
                base_light += 200  # Lights turn on with motion
            
            light_level = max(0, base_light + np.random.normal(0, 50))
            
            # Energy consumption (correlated with activity)
            energy_base = 100 if is_business_hours and weekday else 50
            if temperature > 25:  # AC usage
                energy_base += (temperature - 25) * 20
            elif temperature < 18:  # Heating usage
                energy_base += (18 - temperature) * 15
            
            energy_consumption = max(0, energy_base + np.random.normal(0, 20))
            
            # Add some sensor errors/missing values
            if random.random() < 0.02:  # 2% sensor failure rate
                temperature = None
                humidity = None
            
            data.append({
                'Timestamp': timestamp,
                'Location': location,
                'Sensor_ID': f'{location}_SENSOR_{np.random.randint(1, 6):02d}',
                'Temperature_C': round(temperature, 1) if temperature is not None else None,
                'Humidity_Percent': round(humidity, 1) if humidity is not None else None,
                'Air_Quality_Index': round(air_quality_index, 1),
                'Motion_Detected': motion_detected,
                'Light_Level_Lux': round(light_level, 1),
                'Energy_Consumption_kWh': round(energy_consumption, 2),
                'Battery_Level': max(0, 100 - np.random.exponential(2)),  # Battery degradation
                'Signal_Strength': round(np.random.uniform(-70, -30), 1),  # dBm
                'Status': np.random.choice(['Normal', 'Warning', 'Error'], p=[0.9, 0.08, 0.02])
            })
        
        df = pd.DataFrame(data)
        
        # Add some extreme outliers (sensor malfunctions)
        outlier_indices = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
        df.loc[outlier_indices, 'Temperature_C'] = np.random.uniform(-50, 100, size=len(outlier_indices))
        df.loc[outlier_indices, 'Status'] = 'Error'
        
        return df
    
    def generate_financial_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """
        Generate financial market data.
        
        Args:
            n_rows: Number of trading records to generate
            
        Returns:
            DataFrame with financial data
        """
        
        # Stock symbols and sectors
        stocks = {
            'TECH_001': 'Technology', 'TECH_002': 'Technology', 'TECH_003': 'Technology',
            'HEALTH_001': 'Healthcare', 'HEALTH_002': 'Healthcare',
            'FINANCE_001': 'Finance', 'FINANCE_002': 'Finance', 'FINANCE_003': 'Finance',
            'ENERGY_001': 'Energy', 'ENERGY_002': 'Energy',
            'RETAIL_001': 'Retail', 'RETAIL_002': 'Retail'
        }
        
        # Generate trading days
        start_date = datetime.now() - timedelta(days=n_rows//5)
        trading_dates = []
        current_date = start_date
        while len(trading_dates) < n_rows:
            if current_date.weekday() < 5:  # Only weekdays
                trading_dates.append(current_date)
            current_date += timedelta(days=1)
        
        data = []
        stock_prices = {symbol: np.random.uniform(50, 200) for symbol in stocks.keys()}
        
        for i, date in enumerate(trading_dates):
            symbol = np.random.choice(list(stocks.keys()))
            sector = stocks[symbol]
            
            # Simulate price movement (random walk with drift)
            price_change = np.random.normal(0.001, 0.02)  # Slight upward drift
            stock_prices[symbol] *= (1 + price_change)
            stock_prices[symbol] = max(1, stock_prices[symbol])  # Prevent negative prices
            
            close_price = stock_prices[symbol]
            
            # Generate OHLC data
            volatility = np.random.uniform(0.01, 0.05)
            high = close_price * (1 + np.random.uniform(0, volatility))
            low = close_price * (1 - np.random.uniform(0, volatility))
            open_price = np.random.uniform(low, high)
            
            # Volume influenced by price changes
            base_volume = np.random.randint(100000, 1000000)
            if abs(price_change) > 0.03:  # High volatility increases volume
                base_volume *= 2
            
            volume = int(base_volume * np.random.uniform(0.5, 2))
            
            # Calculate technical indicators
            market_cap = close_price * np.random.randint(10000000, 1000000000)
            pe_ratio = np.random.uniform(10, 30) if random.random() > 0.1 else None
            
            data.append({
                'Date': date,
                'Symbol': symbol,
                'Sector': sector,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close_price, 2),
                'Volume': volume,
                'Market_Cap': market_cap,
                'PE_Ratio': round(pe_ratio, 2) if pe_ratio else None,
                'Price_Change': round(price_change * 100, 2),  # Percentage
                'Volatility': round(volatility * 100, 2),
                'Trading_Halted': np.random.choice([True, False], p=[0.01, 0.99]),
                'Dividend_Yield': round(np.random.uniform(0, 0.05), 4) if random.random() > 0.3 else None
            })
        
        return pd.DataFrame(data)
    
    def save_sample_datasets(self, output_dir: str = "sample_datasets"):
        """
        Generate and save all sample datasets to files.
        
        Args:
            output_dir: Directory to save the datasets
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {
            'sales_data.csv': self.generate_sales_data(1000),
            'customer_data.csv': self.generate_customer_data(500),
            'marketing_campaigns.csv': self.generate_marketing_data(2000),
            'iot_sensor_data.csv': self.generate_iot_sensor_data(5000),
            'financial_data.csv': self.generate_financial_data(1000)
        }
        
        for filename, df in datasets.items():
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"✅ Generated {filename} with {len(df)} rows and {len(df.columns)} columns")
        
        # Also create Excel and JSON versions for testing
        sales_df = datasets['sales_data.csv']
        sales_df.to_excel(os.path.join(output_dir, 'sales_data.xlsx'), index=False)
        
        # Create a subset for JSON (JSON doesn't handle datetime well)
        customer_subset = datasets['customer_data.csv'].head(100)
        customer_subset.to_json(os.path.join(output_dir, 'customer_sample.json'), 
                               orient='records', date_format='iso')
        
        print(f"\n📁 All sample datasets saved to '{output_dir}' directory")
        print("Files include CSV, Excel, and JSON formats for testing different upload types")

if __name__ == "__main__":
    # Generate sample datasets when run directly
    generator = SampleDataGenerator()
    generator.save_sample_datasets()