#!/usr/bin/env python3
"""
Test script to verify that all visualizations work correctly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configure plotly
pio.templates.default = "plotly_white"
pio.renderers.default = "browser"

def test_plotly_charts():
    """Test basic plotly charts"""
    print("Testing Plotly charts...")
    
    # Test data
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10],
        'category': ['A', 'B', 'A', 'B', 'A']
    })
    
    # Test bar chart
    fig_bar = px.bar(data, x='x', y='y', title='Test Bar Chart')
    fig_bar.update_layout(height=400, showlegend=True)
    print("✓ Bar chart created successfully")
    
    # Test line chart
    fig_line = px.line(data, x='x', y='y', title='Test Line Chart')
    fig_line.update_layout(height=400, showlegend=True)
    print("✓ Line chart created successfully")
    
    # Test scatter plot
    fig_scatter = px.scatter(data, x='x', y='y', color='category', title='Test Scatter Plot')
    fig_scatter.update_layout(height=400, showlegend=True)
    print("✓ Scatter plot created successfully")
    
    # Test histogram
    fig_hist = px.histogram(data, x='y', nbins=5, title='Test Histogram')
    fig_hist.update_layout(height=400, showlegend=False)
    print("✓ Histogram created successfully")
    
    # Test gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0.7,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Test Gauge"},
        gauge={'axis': {'range': [None, 1]},
               'bar': {'color': "darkgreen"},
               'steps': [
                   {'range': [0, 0.5], 'color': "lightgray"},
                   {'range': [0.5, 1], 'color': "lightgreen"}
               ]}
    ))
    print("✓ Gauge chart created successfully")
    
    return True

def test_matplotlib_charts():
    """Test basic matplotlib charts"""
    print("\nTesting Matplotlib charts...")
    
    # Test data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    
    # Test basic plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("Test Matplotlib Plot")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.close()
    print("✓ Matplotlib plot created successfully")
    
    return True

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\nTesting dependencies...")
    
    try:
        import streamlit
        print(f"✓ Streamlit version: {streamlit.__version__}")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✓ Plotly version: {plotly.__version__}")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas version: {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✓ Numpy version: {numpy.__version__}")
    except ImportError as e:
        print(f"✗ Numpy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("VISUALIZATION TEST SUITE")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_tests_passed = False
    
    # Test visualizations
    if not test_plotly_charts():
        all_tests_passed = False
    
    if not test_matplotlib_charts():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Your visualizations should work correctly.")
    else:
        print("❌ SOME TESTS FAILED! Check the errors above.")
    print("=" * 50)