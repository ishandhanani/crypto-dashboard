"""
Differential Capital Dashboard
"""

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import sys
from textblob import TextBlob
import tweepy
import matplotlib.pyplot as plt
import re


'''
A function that reads in csv data files that hold OHLCV data for crypto currencies.

Paramters
==========
name: string
    The name of the file
'''


def read_data(name = 'btcusd'):
    path = f"data/{name}.csv"
    df = pd.read_csv(path)    
    x = ['time','open','close','high','low','volume']
    return df[x]

"""
Generates a plotly graph with the ability to plot EMA and Bollinger Bands

Parameters
===========
data:   DataFrame
    A dataframe with OHLCV data and a time stamp
time_frame: str
    The time frame to resample the data at in order to graph.
    Available inputs are '24H', '1W','1M','6M','1Y','ALL'
ema_periods: int
    The exponential moving average periods to graph
bollinger_rate: int
    Rate to calculate bollinger bands 
indicator:  str
    Choose which indicators to plot. Available options
    include 'ALL','EMA','Bollinger's Bands'
"""

def graph_indicators(data, 
                    time_frame = 'ALL',
                    ema_periods = 10, 
                    bollinger_rate = 20, 
                    indicator = 'EMA'):

    # This function converts the timestamp into pandas datetime and set it as index
    def change_date(df):
        df['time']= pd.to_datetime(df['time'],unit = 'ms')
        df = df.set_index('time',drop = True)
        df.sort_index
        return df

    def calculate_ema(prices, days = 10, smoothing=2):
        ema = [0]*(days-1) +[sum(prices[:days]) / days] # First method
        for price in prices[days:]:
            ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
        return ema
    
    def resample(data,resolution ='D'):
        data = data.resample(resolution).mean().dropna()
        return data
    
    def get_sma(prices, rate= bollinger_rate): 
        return prices.rolling(rate).mean()

    # Calculate the bollinger bands (Both upper & lower bonds)
    def get_bollinger_bands(prices, rate=bollinger_rate):
        sma = get_sma(prices, rate)
        two_std = 2*prices.rolling(rate).std()
        bollinger_top = sma + two_std # Calculate top band
        bollinger_below = sma - two_std # Calculate bottom band
        return bollinger_top, bollinger_below

    # This function aggregates the functions above and returns the 
    #   dataframe used to graph. The "resolution" parameter will 
    #   be determined based on "time_frame"
    def process_data(data,resolution = 'D', ema_periods = 10, bollinger_rate = 20):
        # first convert the data
        data = change_date(data)

        #then resample the data
        if resolution != 'Min':
            data = resample(data,resolution)

        # then calculate and stores the EMA
        data['ema'] = calculate_ema(data.close, days= ema_periods )

        # Calculate and store the SMA
        data['sma'] = get_sma(data.close, rate = bollinger_rate)

        # calculate the upper & lower bollinger bands and store it
        bollinger_up, bollinger_down = get_bollinger_bands(prices = data.close,rate = bollinger_rate)
        data['bollinger_up'] = bollinger_up
        data['bollinger_down'] = bollinger_down

        return data

    # Function to create the interactive graph
    def make_graphs(data, columns = ['close']):
        figure = px.line(data_frame = data, y =columns)
        layout = go.Layout(
            plot_bgcolor='#f5f5f5',
            # Font Families
            font_family='Balto',
            font_color='#000000',
            font_size=18,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                )
            )
        )
        # Update options and show plot
        figure.update_layout(layout)
        return figure
    
    def select_indicator(chosen = indicator):
        col_to_graph = ['close']
        if chosen == 'ALL':
            col_to_graph.extend(['ema','bollinger_up','bollinger_down'])
        elif chosen.lower() == 'ema':
            col_to_graph.extend([chosen.lower()])
        elif 'bollinger' in chosen.lower(): 
            col_to_graph.extend(['bollinger_up','bollinger_down'])
        return col_to_graph

    # Output based on time frame selected
    if time_frame == '24H':
        # the number inside tail() is (how many min in 24h) so --> (number of 'resolution' in 'time_frame')
        graph_data = process_data(data,resolution='2min',ema_periods=ema_periods,bollinger_rate=bollinger_rate).tail(1440)
        col_to_graph = select_indicator(chosen = indicator)
        figure = make_graphs(data= graph_data, columns = col_to_graph)
        return figure

    if time_frame == 'ALL':
        graph_data = process_data(data,resolution='D',ema_periods=ema_periods,bollinger_rate=bollinger_rate)
        col_to_graph = select_indicator(chosen = indicator)
        figure = make_graphs(data= graph_data, columns = col_to_graph)
        return figure

    if time_frame == '1W':
        graph_data = process_data(data,resolution='15Min',ema_periods=ema_periods,bollinger_rate=bollinger_rate).tail(672)
        col_to_graph = select_indicator(chosen = indicator)
        figure = make_graphs(data= graph_data, columns =col_to_graph)
        return figure

    if time_frame == '1M':
        graph_data = process_data(data,resolution='3H',ema_periods=ema_periods,bollinger_rate=bollinger_rate).tail(240)
        col_to_graph = select_indicator(chosen = indicator)
        figure = make_graphs(data= graph_data, columns =col_to_graph) 
        return figure

    if time_frame == '6M':
        graph_data = process_data(data,resolution='D',ema_periods=ema_periods,bollinger_rate=bollinger_rate).tail(180)
        col_to_graph = select_indicator(chosen = indicator)
        figure = make_graphs(data= graph_data, columns =col_to_graph)
        return figure

    if time_frame == '1Y':
        graph_data = process_data(data,resolution='D',ema_periods=ema_periods,bollinger_rate=bollinger_rate).tail(365)
        col_to_graph = select_indicator(chosen = indicator)
        figure = make_graphs(data= graph_data, columns = col_to_graph)
        return figure       

    # if something not included gets put iin, just graph the all time data without any indicators
    else:
        graph_data = process_data(data = data, resolution = 'D')
        figure = make_graphs(data = graph_data, columns =['close'])
        return figure


"""
Generates a plotly graph with the ability to plot MACD

Parameters
===========
data:   DataFrame
    A dataframe with OHLCV data and a time stamp
time_frame: str
    The time frame to resample the data at in order to graph.
    Available inputs are '24H', '1W','1M','6M','1Y','ALL'
macd_min: int
    Minimum periods to sample macd 
macd_max: int
    Maximum periods to sample macd 
"""


def graph_macd(data,
                time_frame = 'ALL', 
                macd_min = 12, 
                macd_max = 26):  

    # This function converts the timestamp into pandas datetime and set it as index
    def change_date(df):
        df['time']= pd.to_datetime(df['time'],unit = 'ms')
        df = df.set_index('time',drop = True)
        df.sort_index
        return df

    def resample(data,resolution ='D'):
        data = data.resample(resolution).mean().dropna()
        return data

    def get_macd(prices,min_periods = macd_min, max_periods = macd_max):
        # Get the 26-day EMA of the closing price
        k = prices.ewm(span=12, adjust=False, min_periods=min_periods).mean()
        # Get the 12-day EMA of the closing price
        d = prices.ewm(span=26, adjust=False, min_periods=max_periods).mean()
        #Subtract the 26-day EMA from the 12-Day EMA to get the MACD
        macd = k - d
        # Get the 9-Day EMA of the MACD for the Trigger line
        macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        # Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
        macd_CD = macd - macd_signal 
        return macd, macd_signal, macd_CD

    # This function aggregates the functions above and returns the 
    #   dataframe used to graph. The "resolution" parameter will 
    #   be determined based on "time_frame"
    def process_data(data,resolution = 'D', macd_min = 12, macd_max= 26):
        data = change_date(data)
        data = data.dropna()

        if resolution != 'Min':
            data = resample(data,resolution)

        # calculate the MACD & MACD signal
        macd, macd_signal, macd_CD = get_macd(data.close, min_periods=macd_min, max_periods= macd_max)
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        data['macd_CD'] = macd_CD

        return data

    def make_graphs(data):
        fig = make_subplots(rows=2, cols=1,vertical_spacing = 0.25)
        
        # Price Line
        fig.append_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        line=dict(color='#6495ed', width=1),
        name='closing price',
        # showlegend=False,
        legendgroup='1',
    ), row=1, col=1)
        
        # Candlestick chart for pricing
        fig.append_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                increasing_line_color= 'green',#'#ffa500',
                decreasing_line_color='red',
                showlegend=False
            ), row=1, col=1
        )
        
        # Fast Signal (%k)
        fig.append_trace(
            go.Scatter(
                x=data.index,
                y=data['macd'],
                line=dict(color='#ff7f50', width=2),
                name='MACD',
                #showlegend=False,
                legendgroup='2',
            ), row=2, col=1
        )
        
        # Slow signal (%d)
        fig.append_trace(go.Scatter(
                x=data.index,
                y=data['macd_signal'],
                line=dict(color='#6495ed', width=2),
                #showlegend=False,
                legendgroup='2',
                name='MACD signal'
            ), row=2, col=1
        )
        
        # Colorize the histogram values
        colors = np.where(data['macd_CD'] < 0, '#ff6347', '#00ff00')
        # Plot the histogram
        fig.append_trace(
            go.Bar(
                x=data.index,
                y=data['macd_CD'],
                name='Histogram',
                marker_color=colors,
            ), row=2, col=1
        )
        # Make it pretty
        layout = go.Layout(
            plot_bgcolor='#f5f5f5',
            # Font Families
            font_family='Balto',
            font_color='#000000',
            font_size=20,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                )
            )
        )
        # Update options and show plot
        fig.update_layout(layout)
        return fig

    # dealing with Different Timeframe, assign a resolution to each time frame
    # outputs a plotly express plot

    if time_frame == '24H':
        # the number inside tail() is (how many min in 24h) so --> (number of 'resolution' in 'time_frame')
        graph_data = process_data(data,resolution='2min').tail(1440)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == 'ALL':
        graph_data = process_data(data,resolution='D')
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1W':
        graph_data = process_data(data,resolution='15Min').tail(672)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1M':
        graph_data = process_data(data,resolution='3H').tail(240)
        figure = make_graphs(graph_data)  
        return figure

    if time_frame == '6M':
        graph_data = process_data(data,resolution='D').tail(180)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1Y':
        graph_data = process_data(data,resolution='D').tail(365)
        figure = make_graphs(graph_data)
        return figure       

    # If something not included gets put iin, just graph the all time data 
    #   without any indicators
    else:
        graph_data = process_data(data = data, resolution = 'D')
        figure = px.line(data_frame=graph_data, y = ['close'] )
        layout = go.Layout(
            plot_bgcolor='#f5f5f5',
            # Font Families
            font_family='Balto',
            font_color='#000000',
            font_size=18,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                )
            )
        )
        # Update options and show plot
        figure.update_layout(layout)
        return figure    


"""
Generates a plotly candlestick graph with only SMA plotted
TODO: See if this function can be combined with graph_indicators 
    in order to save render time

Parameters
===========
data:   DataFrame
    A dataframe with OHLCV data and a time stamp
time_frame: str
    The time frame to resample the data at in order to graph.
    Available inputs are '24H', '1W','1M','6M','1Y','ALL'
sma_rate_1: int
    Specified simple moving average rate
sma_rate_2: int
    Specified simple moving average rate 
"""

def graph_candlestick(data,time_frame = 'ALL', sma_rate_1 = 20, sma_rate_2 = 50 ):
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from datetime import datetime, date
    import plotly.graph_objects as go
   
    # This function converts the timestamp into pandas datetime and set it as index
    def change_date(df):
        df['time']= pd.to_datetime(df['time'],unit = 'ms')
        df = df.set_index('time',drop = True)
        df.sort_index
        return df

    # this function resamples the data into sepcified resolution
    def resample(data,resolution ='D'):
        data = data.resample(resolution).mean().dropna()
        return data
    
    # this function calculates SMA based on given rate
    def get_sma(prices, rate= 20): 
        return prices.rolling(rate).mean()

    # this function will return a DataFrame with specified parameters,
    # later the data produced will be used to graph
    # The "resolution" parameter will be determined based on "time_frame"
    def process_data(data,resolution = 'D', sma_rate_1 = 20, sma_rate_2 = 50):
        # first convert the data
        data = change_date(data)
        data = data.dropna()

        #then resample the data
        # if the resolution is not already "Min"
        if resolution != 'Min':
            data = resample(data,resolution)

        # calculate the SMA with different periods
        data['sma_1'] = get_sma(data.close,rate = sma_rate_1)
        data['sma_2'] = get_sma(data.close,rate = sma_rate_2)
        
        # return the DataFrame with all data processed, and Includes all Tech Indicators calculated based on
        # specified resolution 
        return data

    def make_graphs(data,sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2):
        import numpy as np
        import plotly.express as px
        from datetime import datetime, date
        import plotly.graph_objects as go
        
        # create the graph
        fig = go.Figure()

        fig.add_trace(go.Candlestick(x=data.index,
                         open=data['open'],
                         high=data['high'],
                         low=data['low'],
                         close=data['close'],
                         showlegend=False
                                     )
         )

        fig.add_trace(
            {'x': data.index,
            'y': data.sma_1,
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 1,
                'color': '#6495ed'
                    },
            'name': f'SMA of {sma_rate_1} periods'   
            }

        )

        fig.add_trace(
            {'x': data.index,
            'y': data.sma_2,
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 1,
                'color': 'blueviolet'
                    },
            'name': f'SMA of {sma_rate_2} periods'   
            }

        )

       
        # Make it pretty
        layout = go.Layout(
            plot_bgcolor='#f5f5f5',
            # Font Families
            font_family='Balto',
            font_color='#000000',
            font_size=13,
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                )
            )
        )
        # Update options and show plot
        fig.update_layout(layout)
        return fig

    if time_frame == '24H':
        # the number inside tail() is (how many min in 24h) so --> (number of 'resolution' in 'time_frame')
        graph_data = process_data(data,resolution='2Min',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(1440)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == 'ALL':
        graph_data = process_data(data,resolution='D',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1W':
        graph_data = process_data(data,resolution='15Min',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(672)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1M':
        graph_data = process_data(data,resolution='3H',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(240)
        figure = make_graphs(graph_data)  
        return figure

    if time_frame == '6M':
        graph_data = process_data(data,resolution='D',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(180)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1Y':
        graph_data = process_data(data,resolution='D',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(365)
        figure = make_graphs(graph_data)
        return figure       

    # if something not included gets put in, just graph the all time data 
        # without any indicators
    else:
        graph_data = process_data(data = data, resolution = 'D')
        figure = px.line(data_frame=graph_data, y = ['close'] )
        layout = go.Layout(
            plot_bgcolor='#f5f5f5',
            # Font Families
            font_family='Balto',
            font_color='#000000',
            font_size=13,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                )
            )
        )
        # Update options and show plot
        figure.update_layout(layout)
        return figure    


"""
Generates a plotly line graph with only SMA plotted
TODO: See if this function can be combined with graph_indicators 
    in order to save render time

Parameters
===========
data:   DataFrame
    A dataframe with OHLCV data and a time stamp
time_frame: str
    The time frame to resample the data at in order to graph.
    Available inputs are '24H', '1W','1M','6M','1Y','ALL'
sma_rate_1: int
    Simple moving average 
sma_rate_2: int
    Maximum periods to sample macd 
"""


def graph_line_toggle(data,time_frame = 'ALL', sma_rate_1 = 20, sma_rate_2 = 50):
    
    # This function converts the timestamp into pandas datetime and set it as index
    def change_date(df):
        df['time']= pd.to_datetime(df['time'],unit = 'ms')
        df = df.set_index('time',drop = True)
        df.sort_index
        return df

    # this function resamples the data into sepcified resolution
    def resample(data,resolution ='D'):
        data = data.resample(resolution).mean().dropna()
        return data
    
    # this function calculates SMA based on given rate
    def get_sma(prices, rate= 20): 
        return prices.rolling(rate).mean()

    # this function will return a DataFrame with specified parameters,
    # later the data produced will be used to graph
    # The "resolution" parameter will be determined based on "time_frame"
    def process_data(data,resolution = 'D', sma_rate_1 = 20, sma_rate_2 = 50):
        # first convert the data
        data = change_date(data)

        #then resample the data
        # if the resolution is not already "Min"
        if resolution != 'Min':
            data = resample(data,resolution)

        # calculate the SMA with different periods
        data['sma_1'] = get_sma(data.close,rate = sma_rate_1)
        data['sma_2'] = get_sma(data.close,rate = sma_rate_2)
        
        return data

    def make_graphs(data,sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2):
        # create the graph
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index,
                                 y=data['close'],
                                 showlegend=False
                                     )
         )  
    
        fig.add_trace(
            {'x': data.index,
            'y': data.sma_1,
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 1,
                'color': '#6495ed'
                    },
            'name': f'SMA of {sma_rate_1} periods'   
            }

        )

        fig.add_trace(
            {'x': data.index,
            'y': data.sma_2,
            'type': 'scatter',
            'mode': 'lines',
            'line': {
                'width': 1,
                'color': 'blueviolet'
                    },
            'name': f'SMA of {sma_rate_2} periods'   
            }

        )
       
        # Make it pretty
        layout = go.Layout(
            plot_bgcolor='#f5f5f5',
            # Font Families
            font_family='Balto',
            font_color='#000000',
            font_size=13,
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                )
            )
        )
        # Update options and show plot
        fig.update_layout(layout)
        return fig

    # dealing with Different Timeframe, assign a resolution to each time frame

    if time_frame == '24H':
        # the number inside tail() is (how many min in 24h) so --> (number of 'resolution' in 'time_frame')
        graph_data = process_data(data,resolution='2Min',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(1440)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == 'ALL':
        graph_data = process_data(data,resolution='D',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1W':
        graph_data = process_data(data,resolution='15Min',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(672)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1M':
        graph_data = process_data(data,resolution='3H',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(240)
        figure = make_graphs(graph_data)  
        return figure

    if time_frame == '6M':
        graph_data = process_data(data,resolution='D',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(180)
        figure = make_graphs(graph_data)
        return figure

    if time_frame == '1Y':
        graph_data = process_data(data,resolution='D',sma_rate_1 = sma_rate_1, sma_rate_2 = sma_rate_2).tail(365)
        figure = make_graphs(graph_data)
        return figure       

    # if something not included gets put iin, just graph the all time data 
        # without any indicators
    else:
        graph_data = process_data(data = data, resolution = 'D')
        figure = px.line(data_frame=graph_data, y = ['close'] )
        layout = go.Layout(
            plot_bgcolor='#f5f5f5',
            # Font Families
            font_family='Balto',
            font_color='#000000',
            font_size=13,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                )
            )
        )
        # Update options and show plot
        figure.update_layout(layout)
        return figure    


"""
Generates the return on investment given the time period 

Parameters
===========
data:   DataFrame
    A dataframe with OHLCV data and a time stamp
time_frame: str
    The time frame to resample the data at in order to graph.
    Available inputs are '24H', '1W','1M','6M','1Y','ALL'
"""

def calculate_return(data,time_frame = 'ALL'):

    # This function converts the timestamp into pandas datetime and set it as index
    def change_date(df):
        df['time']= pd.to_datetime(df['time'],unit = 'ms')
        df = df.set_index('time',drop = True)
        df.sort_index
        return df

    # this function resamples the data into sepcified resolution
    def resample(data,resolution ='D'):
        data = data.resample(resolution).mean()
        data = data.dropna()
        return data
    
    # this function will return a DataFrame with specified parameters,
    # later the data produced will be used to graph
    # The "resolution" parameter will be determined based on "time_frame"
    def process_data(data,resolution = 'D'):
        # first convert the data
        data = change_date(data)

        #then resample the data
        # if the resolution is not already "Min"
        if resolution != 'Min':
            data = resample(data,resolution)

        # return the DataFrame with all data processed, and Includes all Tech Indicators calculated based on
        # specified resolution 
        return data
    
    # function to calculate the return rate  
    def calculate(begin, end):
        percent_change = ((end-begin)/begin)*100
        return f"{round(percent_change[0],2)}%"

    # dealing with Different Timeframe, assign a resolution to each time frame
    if time_frame == '24H':
        # the number inside tail() is (how many min in 24h) so --> (number of 'resolution' in 'time_frame')
        selected_data = process_data(data,resolution='Min').tail(1440)
        num = calculate(begin = selected_data.head(1).close.values, end = selected_data.tail(1).close.values)
        return num

    if time_frame == 'ALL':
        selected_data = process_data(data,resolution='D')
        num = calculate(begin = selected_data.head(1).close.values, end = selected_data.tail(1).close.values)
        return num


    if time_frame == '1W':
        selected_data = process_data(data,resolution='Min').tail(672)
        num = calculate(begin = selected_data.head(1).close.values, end = selected_data.tail(1).close.values)
        return num

    if time_frame == '1M':
        selected_data = process_data(data,resolution='3H').tail(240)
        num = calculate(begin = selected_data.head(1).close.values, end = selected_data.tail(1).close.values)
        return num

    if time_frame == '6M':
        selected_data = process_data(data,resolution='D').tail(180)
        num = calculate(begin = selected_data.head(1).close.values, end = selected_data.tail(1).close.values)
        return num

    if time_frame == '1Y':

        selected_data = process_data(data,resolution='D').tail(365)
        num = calculate(begin = selected_data.head(1).close.values, end = selected_data.tail(1).close.values)
        return num


"""
Generates a processed dataframe resampled according to the time frame given.
#TODO: This is the exact code that is used to start the other functions. Create a class
    that has this boilerplate so it's not copied so many times

Parameters
===========
data:   DataFrame
    A dataframe with OHLCV data and a time stamp
time_frame: str
    The time frame to resample the data at in order to graph.
    Available inputs are '24H', '1W','1M','6M','1Y','ALL'
"""


def correl_plot(data,time_frame = 'ALL'):
    # This function converts the timestamp into pandas datetime and set it as index
    def change_date(df):
        df['time']= pd.to_datetime(df['time'],unit = 'ms')
        df = df.set_index('time',drop = True)
        df.sort_index
        return df

    # this function resamples the data into sepcified resolution
    def resample(data,resolution ='D'):
        data = data.resample(resolution).mean()
        data = data.dropna()
        return data

    # this function will return a DataFrame with specified parameters,
    # later the data produced will be used to graph
    # The "resolution" parameter will be determined based on "time_frame"
    def process_data(data,resolution = 'D'):
        # first convert the data
        data = change_date(data)

        #then resample the data
        # if the resolution is not already "Min"
        if resolution != 'Min':
            data = resample(data,resolution)

        # return the DataFrame with all data processed
        return data

    if time_frame == '24H':
        # the number inside tail() is (how many min in 24h) so --> (number of 'resolution' in 'time_frame')
        graph_data = process_data(data,resolution='2Min').tail(1440)
        return graph_data

    if time_frame == 'ALL':
        graph_data = process_data(data,resolution='D')
        return graph_data

    if time_frame == '1W':
        graph_data = process_data(data,resolution='15Min').tail(672)
        return graph_data

    if time_frame == '1M':
        graph_data = process_data(data,resolution='3H').tail(240)  
        return graph_data

    if time_frame == '6M':
        graph_data = process_data(data,resolution='D').tail(180)
        return graph_data

    if time_frame == '1Y':
        graph_data = process_data(data,resolution='D').tail(365)
        return graph_data 


"""
Generates a dataframe that gives percent change for the close column. It's
used to create the correlation plot between returns on the statistics page

Parameters
===========
data:   DataFrame
    A dataframe with OHLCV data and a time stamp
time_frame: str
    The time frame to resample the data at in order to graph.
    Available inputs are '24H', '1W','1M','6M','1Y','ALL'
"""

def graph_returns(data, tf):
    df = correl_plot(data,tf)['close'].to_frame()
    df['return'] = np.NaN
    returns = df.pct_change()
    df['return'] = returns['close']
    df = df.fillna(0)
    return df

"""
Functions for the sentiment analysis page
#TODO: Refactor this code to make it more modular
        Add support and filter different phrases automatically (crypto specific)
"""

#API access for Tweepy
Consumer_Key = 'ocaT3ODhqPQ0oSCP8nbO9DIZt' 
Consumer_Secret = 'Z8C5rnrVuMwFjaOMxSm8R71PZX1AnaiJdryFjm2HKJ11PbbqnI'
Access_Token = '1508668306965901320-0VMQQQThgZiPKTtgvBlkuGBWO9Zjsd'
Access_Token_Secret = 'jJX9us34oDBIzxgUNW9DyqbBLpfLToOlaaoMsX7LkeLBk'
my_bearer_token = 'AAAAAAAAAAAAAAAAAAAAALGNawEAAAAAwgE3kxG%2BC5IPiFADuwL4xMImTJc%3DJ67OT8NBhIQMqlWA6thvycOlGBtIN9TH2lAeJkE57faRXPqQRY'
client = tweepy.Client(bearer_token = my_bearer_token)

Authorize = tweepy.OAuthHandler(Consumer_Key, Consumer_Secret)
Authorize.set_access_token(Access_Token,Access_Token_Secret)
api = tweepy.API(Authorize)

def Percentage(element, total): #Simple Percent Function
    P = 100*float(element)/(total)
    return round(P,2)


def clean_tweet(tweet):
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    temp = tweet.lower()
    # To avoid removing contractions in english
    temp = re.sub("'", "", temp) 
    temp = re.sub("@[A-Za-z0-9_]+","", temp) 
    # Next 6 lines remove links, @'s to other users and other symbols from tweets'
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    # Does not include extraneous words
    temp = [w for w in temp if not w in stopwords] 
    temp = " ".join(word for word in temp)
    return temp

def Day_Sentiment(Word):
    query = Word
    #API Request
    Messages = client.search_recent_tweets(query=query, max_results = 100) 
    Messages = str(Messages)
    #Splits each tweet
    List_Messages = Messages.split('<Tweet id') 
    
    Positive = 0
    Negative = 0
    for i in List_Messages:
        cleaned = clean_tweet(str(i))
        Analysis = TextBlob(cleaned) #pass each element of tweet list as string, TextBlob performs sentiment analysis
        Polarity = Analysis.sentiment.polarity #Polarity (how positive or negative) each message is
        # Neutral sentiment
        if (Polarity == 0): 
            pass
        # Negative sentiment
        elif (Polarity < 0): 
            Negative += Analysis.sentiment.polarity*Analysis.sentiment.subjectivity 
        # Positive sentiment 
        elif (Polarity > 0):
            Positive += (-1)*Analysis.sentiment.polarity*(1-Analysis.sentiment.subjectivity)
       
    Sum = Positive+Negative
    Positive_Percentage = Percentage(Positive,Sum)
    Negative_Percentage = Percentage(Negative,Sum)
   
    labels = ['Positive ['+str(Positive_Percentage)+'%]' , 'Negative ['+str(Negative_Percentage)+'%]'] #Making plot
    sizes = [Positive_Percentage, Negative_Percentage]

    x = ['Positive','Negative']
    df = pd.DataFrame()
    df['Percentage'] = sizes
    df['type'] = x
    fig = px.pie(df, values = sizes, names= x, color_discrete_sequence=["Blue", "Purple"])
    return fig

"""
Code to render the top bar of the dashboard. Data is required to be in the /data folder
All of the cryptos can be replaced with different ones if data is given. Make sure to change
#TODO: Generalize so names don't have to be changed everytime new file is given
"""
# variables holding current price
btc = read_data('btcusd').tail(2).iloc[1,3]
eth = read_data('ethusd').tail(2).iloc[1,3]
#bnb = 425.22 #read_data('').tail(2).iloc[1,3]
#xrp = read_data('xrpusd').tail(2).iloc[1,3]
#ada = read_data('adausd').tail(2).iloc[1,3]
sol = read_data('solusd').tail(2).iloc[1,3]
#terra = read_data('terraust-usd').tail(2).iloc[1,3]
#avax = read_data('btcusd').tail(2).iloc[1,3]

#div elements for displaying 
btcd=html.Div(children = html.Div(f'{btc:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})
ethd=html.Div(children = html.Div(f'{eth:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})
#bnbd=html.Div(children = html.Div(f'{bnb:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})
#xrpd=html.Div(children = html.Div(f'{xrp:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})
#adad=html.Div(children = html.Div(f'{ada:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})
sold=html.Div(children = html.Div(f'{sol:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})
#terrad=html.Div(children = html.Div(f'{terra:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})
#avaxd=html.Div(children = html.Div(f'{avax:.2f}',style={'font-size':'20px','font-family':'sans-serif'}), style={'color': 'white', 'backgroundColor': '#cfc16b'})

colors = {                                
    'background': '#6baaa6',
    'text': '#7FDBFF'
}

"""
Dashboard physical outline
"""

app = dash.Dash(external_stylesheets=[dbc.themes.YETI])
server = app.server

#the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "19rem",
    "padding": "1rem 1rem",
    "background-color": "#6baaa6",
}

#the styles for the main content position
CONTENT_STYLE = {
    "margin-left": "19rem",
    "margin-right": "1rem",
    "padding": "1rem 1rem",
}

#this is the control panel of the main page (quick look candlestick)
controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Crypto Currency"),
                #dbc.Input(id = "chosen_currency_candlestick", placeholder = 'Choose a currency', value ='BTCUSD', type = 'text'),
                dcc.Dropdown(
                    id="chosen_currency_candlestick",
                    options=
                        [
                           {'label':c, 'value':c} for c in ['BTCUSD', 'ETHUSD','SOLUSD']
                        ],
                    value="BTCUSD",
                    clearable = False,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Timeframe shown:"),
                dcc.Dropdown(
                    id="time_frame_candlestick",
                    options=
                        [
                            {'label':c,'value':c} for c in ['ALL','1Y','6M','1M','1W','24H']
                        ],
                    value="24H",
                    clearable = False,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Toggle Line: "),
                dcc.Dropdown(
                    id='toggle',
                    options=
                        [
                            {"label": c, "value": c} for c in ['Candlestick', 'Line']
                        ],
                    value='Candlestick',
                    clearable = False
                )
            ]
        )
    ],body=True
)
#this is the control panel for the statistics page
controls_stat = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Crypto Currency"),
                dcc.Dropdown(
                    id="chosen_currency_stat",
                    options=
                        [
                           {'label':c, 'value':c} for c in ['BTCUSD', 'ETHUSD','SOLUSD']
                        ],
                    value=['BTCUSD', 'ETHUSD'],
                    multi=True,
                    clearable = False,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Timeframe shown:"),
                dcc.Dropdown(
                    id="time_frame_stat",
                    options=
                        [
                            {'label':c,'value':c} for c in ['ALL','1Y','6M','1M','1W']
                        ],
                    value="1M",
                    clearable = False,
                ),
            ]
        ),
    ],body=True
)

#this is the control panel for the sentiment page
controls_sent = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Crypto Currency"),
                dbc.Input(id="input_sent", placeholder="Type something...", type="text")
            ]
        ),
    ],body=True
)

# this is the control panel of the indicator page (candlestick)
controls_indicators = dbc.Card(  
    [
        html.Div(
            [
                dbc.Label("Crypto Currency"),
                #dbc.Input(id = "chosen_currency_indicators", placeholder = 'Choose a currency', value ='BTCUSD', type = 'text'),
                dcc.Dropdown(
                    id="chosen_currency_indicators",
                    options=
                    [
                       {'label':c, 'value':c} for c in ['BTCUSD', 'ETHUSD','SOLUSD']
                    ],
                    value="BTCUSD",
                    clearable = False,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Timeframe shown:"),
                dcc.Dropdown(
                    id="time_frame_indicators",
                    options=
                        [
                            {'label':c,'value':c} for c in ['ALL','1Y','6M','1M','1W','24H']
                        ],
                    value="24H",
                    clearable = False,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Indicator to graph:"),
                dcc.Dropdown(
                    id="chosen_indicators",
                    options=
                        [
                            {'label':c,'value':c} for c in ['ALL', 'EMA',"Bollinger's Bands",'MACD']
                        ],
                    value="EMA",
                    clearable = False,
                ),
            ]
        ),
    ],body=True
)

# this is the control panel of the MACD graph
controls_macd = dbc.Card(  
    [
        html.Div(
            [
                dbc.Label("Crypto Currency"),
                #dbc.Input(id = "chosen_currency_macd", placeholder = 'Choose a currency', value ='BTCUSD', type = 'text'),
                dcc.Dropdown(
                    id="chosen_currency_macd",
                    options=
                        [
                           {'label':c, 'value':c} for c in ['BTCUSD', 'ETHUSD','SOLUSD']
                        ],
                    value="BTCUSD",
                    clearable = False,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Timeframe shown:"),
                dcc.Dropdown(
                    id="time_frame_macd",
                    options=
                        [
                            {'label':c,'value':c} for c in ['ALL','1Y','6M','1M','1W','24H']
                        ],
                    value="24H",
                    clearable = False,
                ),
            ]
        ),
    ],body=True
)

#This creates the sidebar of the dashboard
sidebar = html.Div(
    [
        html.P("Crypto Dashboard", className="display-4"),
        html.Hr(),
        html.P("A cryptocurrency asset dashboard for Differential Capital", 
               className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Stats", href="/stats", active="exact"),
                dbc.NavLink("Sentiment Analysis", href="/sentiment", active="exact"),
                dbc.NavLink("Technical Indicators", href="/charting", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],style=SIDEBAR_STYLE
)

#this is the container for all the content that goes on the page
content = html.Div(id="page-content", style=CONTENT_STYLE)

#initializing the app
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


"""
Callbacks for initializing the pages
"""


# Callback for creating the pages
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    #initializing the main page
    if pathname == "/":   
        return dbc.Container(
            [
                #below is the code for the top bar
                html.Div(
                    style={'backgroundColor': colors['background'], 'height':'125px'},
                    children=
                        [
                            html.Div(children=[html.Div('BTC', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), btcd], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px', 'margin-left':'25px'}),
                            html.Div(children=[html.Div('ETH', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), ethd], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px'}),
                            #html.Div(children=[html.Div('BNB', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), bnbd], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px'}),
                            #html.Div(children=[html.Div('XRP', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), xrpd], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px'}),
                            #html.Div(children=[html.Div('ADA', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), adad], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px'}),
                            html.Div(children=[html.Div('SOL', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), sold], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px'}),
                            #html.Div(children=[html.Div('TERRA', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), terrad], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px'}),
                            #html.Div(children=[html.Div('AVAX', style={'color':'white','margin-top':'25px','margin-bottom':'25px','backgroundColor': '#297353', 'font-size':'20px','font-family':'sans-serif'}), avaxd], style={'width': '11%', 'display': 'inline-block', 'text-align':'center','margin-right': '10px'})
                        ]
                ),
                #initializing the rest of the page
                html.H1("Quick look Candlestick graph"),
                html.Hr(),
                dbc.Col(
                    [
                        dbc.Col(controls, md=2),
                        dbc.Col(dcc.Graph(id="candlestick")),
                        dbc.Row(html.Div(id= 'table_container')),
                    ],align="center"
                )
            ],fluid=True
        )
    
    #initializing the stats page
    elif pathname == "/stats":
        return dbc.Container(
            [
                html.H1("Statistics"),
                html.P("Use this page to view correlation between assets and information about each coin"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(controls_stat, md=2),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="corr_graph")),
                                dbc.Col(dcc.Graph(id="return_graph"))
                            ]
                        )

                    ],align="center"
                )
            ],fluid=True
        )
    
    #initializing the sentiment analysis page    
    elif pathname == "/sentiment":
            return dbc.Container(
            [
                html.H1("Sentiment Analysis"),
                html.P("Use this page to view the Twitter sentiment behind each coin"),
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(controls_sent, lg=5),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="sent_pie")),
                            ]
                        )

                    ],align="center"
                )
            ],fluid=True
        )
    
    #initializing the TA page
    elif pathname == "/charting":   # content for the tech indicators page
        return dbc.Container(
            [
                html.H1("Technical Indicators Page"),
                html.Hr(),
                dbc.Col(
                    [
                        dbc.Col(controls_indicators, md=2),
                        dbc.Col(dcc.Graph(id="indicators")),
                        dbc.Row(html.Div(id= 'table_container_indicators')),
                    ],align="center"
                ),
            ],fluid=True
        )
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


"""
Callback for home page
"""


#updates the candlestick graph
@app.callback(   # this callback updates the candlestick graph
    Output('candlestick','figure'),
    [Input("chosen_currency_candlestick",'value'),
     Input("time_frame_candlestick",'value'),
     Input('toggle','value')]
)
# the function for graph_candlestick graph
def update_candlestick(chosen_currency_candlestick, time_frame_candlestick, toggle):
    df = read_data(chosen_currency_candlestick.lower())
    time_frame = time_frame_candlestick
    return graph_candlestick(df,time_frame) if toggle == 'Candlestick' else graph_line_toggle(df,time_frame)

@app.callback(     # this callback updates the table of contents(return for the candlestick page)
    Output('table_container','children'),
    [Input('chosen_currency_candlestick','value'),
     Input('time_frame_candlestick','value')]
)

def update_table(chosen_currency_candlestick, time_frame_candlestick):  # creae a dataframe with contents such as return rate
    df = read_data(chosen_currency_candlestick.lower())
    percent_change = calculate_return(df, time_frame_candlestick)
    df = pd.DataFrame(
        {
            'content':['Return within period:'],
            'value':[percent_change]
        }
    )
    table = dbc.Table.from_dataframe(df,striped=True, bordered=True, hover=True)  # the dash-boostrap-component conviently converts the data frame into heml table
    return table


"""
Callbacks for Statistics page
#TODO: Most of this page was hard coded in and functions were not created
        Create functions to store the update correlation and retuns for cleanliness
"""

@app.callback( #updating the heatmap
     Output('corr_graph', 'figure'),
     [Input("chosen_currency_stat",'value'),
      Input("time_frame_stat",'value')]
)
def update_corr(chosen_currency_stat, time_frame_stat):
    chosen_currency_stat = [x.lower() for x in chosen_currency_stat]   
    master_corr_df = pd.DataFrame() #empty dataframe to store the closing prices of each coin
    time_frame = time_frame_stat
    ct = 0
    for val in chosen_currency_stat:
        if ct == 0:
            df = read_data(val)
            dff = correl_plot(df,time_frame)['close']
            master_corr_df = dff
            master_corr_df.reset_index(drop=True,inplace=True)
            ct = 1
        else:
            df = read_data(val)
            dff = correl_plot(df,time_frame)['close']
            dff.reset_index(drop=True,inplace=True)
            diff = len(master_corr_df) - len(dff)
            if diff == 0:
                master_corr_df = pd.concat([master_corr_df, dff], axis=1)
            if diff > 0:
                master_corr_df = master_corr_df.iloc[abs(diff):]
                master_corr_df.reset_index(drop=True,inplace=True)
                master_corr_df = pd.concat([master_corr_df, dff], axis=1)   

            if diff < 0:
                dff = dff.iloc[abs(diff):]
                dff.reset_index(drop=True,inplace=True)
                master_corr_df = pd.concat([master_corr_df, dff], axis=1) 
            
            
    master_corr_df.columns = [chosen_currency_stat]
    testc = master_corr_df.corr()

    return px.imshow(testc.values, color_continuous_scale='BuPu', text_auto=True)

@app.callback( #updating the return graph
     Output('return_graph', 'figure'),
     [Input("chosen_currency_stat",'value'),
      Input("time_frame_stat",'value')]
)

def update_return(chosen_currency_stat,time_frame_stat):
    chosen_currency_stat = [x.lower() for x in chosen_currency_stat]   
    master_corr_df = pd.DataFrame()
    ct = 0
    for val in chosen_currency_stat:
        if ct == 0:
            df = read_data(val)
            dff = graph_returns(df,time_frame_stat)['return']
            master_corr_df = dff.to_frame()
            master_corr_df.reset_index(drop=False,inplace=True)
            ct = 1
        else:
            df = read_data(val)
            dff = graph_returns(df,time_frame_stat)['return']
            dff.reset_index(drop=True,inplace=True)
            diff = len(master_corr_df) - len(dff)
            if diff == 0:
                master_corr_df = pd.concat([master_corr_df, dff], axis=1)
            if diff > 0:
                master_corr_df = master_corr_df.iloc[abs(diff):]
                master_corr_df.reset_index(drop=True,inplace=True)
                master_corr_df = pd.concat([master_corr_df, dff], axis=1)   
            if diff < 0:
                dff = dff.iloc[abs(diff):]
                dff.reset_index(drop=True,inplace=True)
                print(len(dff))
                master_corr_df = pd.concat([master_corr_df, dff], axis=1) 

    master_corr_df.reset_index(drop=True, inplace=True)
    master_corr_df.set_index('time', inplace=True) 
    master_corr_df.index.name = None
    master_corr_df.columns = [chosen_currency_stat]

    # create the graph
    # TODO: Add a title to this graph
    fig = go.Figure()
    for i in range(0,len(chosen_currency_stat)): 
        fig.add_trace(
            {'x': master_corr_df.index,
            'y': master_corr_df.iloc[:,i],
            'type': 'scatter',
            'mode': 'lines',
            'name': str(master_corr_df.columns[i])
            }
        )

    # Make it pretty
    layout = go.Layout(
        plot_bgcolor='#f5f5f5',
        # Font Families
        font_family='Balto',
        font_color='#000000',
        font_size=13,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )
    # Update options and show plot
    fig.update_layout(layout)
    return fig

"""
Callbacks for Sentiment analysis page
"""


@app.callback( #updating the pie chart based on the selected value
     Output('sent_pie', 'figure'),
     [Input("input_sent",'value')]
)
#the function that updates
def update_pie(chosen_currency_sent):
    w = str(chosen_currency_sent)
    return Day_Sentiment(w)

"""
Callbacks for technical indicators page
"""

@app.callback(   # this callback updates the INDICATORS graph
    Output('indicators','figure'),
#    Output('return_candle','children'),
    [Input("chosen_currency_indicators",'value'),
     Input("time_frame_indicators",'value'),
    Input("chosen_indicators",'value')],
)
# the function for graph_candlestick graph
def update_indicators(chosen_currency_indicators, time_frame_indicators,chosen_indicators):
    df = read_data(chosen_currency_indicators.lower())
    time_frame = time_frame_indicators
    if chosen_indicators != 'MACD':
        figure = graph_indicators(df,time_frame,indicator = chosen_indicators)
    else:
        figure = graph_macd(df,time_frame)
    return figure

@app.callback(     # this callback updates the table of contents(return for the indicators page)
    Output('table_container_indicators','children'),
    [Input('chosen_currency_indicators','value'),
     Input('time_frame_indicators','value')]
)

def update_table_indicators(chosen_currency_indicators, time_frame_indicators):  # creae a dataframe with contents such as return rate
    df = read_data(chosen_currency_indicators.lower())
    percent_change = calculate_return(df, time_frame_indicators)
    df = pd.DataFrame(
        {
            'content':['Return within period:'],
            'value':[percent_change]
        }
    )
    table = dbc.Table.from_dataframe(df,striped=True, bordered=True, hover=True)  # the dash-boostrap-component conviently converts the data frame into heml table
    return table


"""
Initialize dashboard on port 8910
"""

if __name__ == "__main__":
    #app.run_server(port=8910 ,debug=True) #local running code
    app.run_server(host='0.0.0.0', port=8050, debug=True)


