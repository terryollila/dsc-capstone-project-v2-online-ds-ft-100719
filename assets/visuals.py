import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import seaborn as sns
import pandas as pd
import numpy as np
import base64

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from dash.dependencies import Input, Output
from wordcloud import WordCloud
from PIL import Image
from os import path
from jupyter_plotly_dash import JupyterDash

import functions as fun

rotten_df_cut = pd.read_csv('rotten_df_cut.csv', index_col=0)
screenplays_cut = pd.read_csv('screenplays_cut.csv', index_col=0)

def cut_down_dist(data1, data2, std_level, label_1, label_2):
    std = data1.mean() \
        + data1.std()*std_level
    
    plot_info_1 = data1.drop(data1[lambda x: x > std].index)
    
    std2 = data2.mean() \
        + data2.std()*std_level
    
    plot_info_2 = data2.drop(data2[lambda x: x > std].index)

    fig = plt.figure(figsize=(10,8))
    sns.distplot(plot_info_1, label=label_1, bins=100, norm_hist=False)
    sns.distplot(plot_info_2, label=label_2, bins=100, norm_hist=False)
    plt.legend()
    plt.show();

def top_words(words, max_features, min_df, max_df):
    """Takes in a series of documents and returns an ordered list of 
    how frequently words appear as calculated by sum vs count.
    
    Parameters:
    
        words: Series
            A series of documents with words to be counted.
            
        max_features: int
            Populates max_features value in vectorizer. Ceiling for how many
            words to use.
            
        min_df: float or int
            Populates min_df value in vectorizer. Minimum documents a word
            must appear in to be counted.
            
        max_df: float or int
            Populates max_df value in vectorizer. Maximum documents a word
            must appear in to be counted.
        
    returns: 
        List of tuples with word and ratio calculated by sum / count:
        (word, ratio), sorted by ratio."""
    
    # Initialize vectorizor and fit
    victor = CountVectorizer(max_features=max_features, 
                             min_df=min_df, max_df=max_df)
    movies_victor = victor.fit_transform(words)
    
    # Transform into SparceDataFrame.
    sdf = pd.SparseDataFrame(movies_victor, 
                                     columns=victor.get_feature_names())
    
    sdf.fillna(0, inplace=True)
    
    # Ave_word_count will house the tuples data to be sorted.
ave_word_count = []
for col in sdf.columns:
    key = col
        
    # Calculate the ratio and add tuple to list.
    value = sum(sdf[col]) / len(sdf[col])
    ave_word_count.append((key, value))
        
    # Return sorted tuple with word and ratio.
    return sorted(ave_word_count, key=lambda x: x[1], reverse=True)

POS_abb = screenplays_cut.columns[list(screenplays_cut.columns).index(
    'word_count'):
                                  list(screenplays_cut.columns).index(
                                      'sentence_length')+1]
POS_abb = POS_abb.append(screenplays_cut.columns[list(screenplays_cut.columns).index(
    'PROPN'):
                                   list(screenplays_cut.columns).index(
                                       'PRON')+1])
POS_abb = list(POS_abb)
POS_abb.remove('temp')
POS_abb.remove('sentiment_scores')

# List of dictionaries for script attribute dropdown
POS_desc = ['WORD COUNT', 'UNIQUE WORDS', 'NEGATIVE SENTIMENT', 
            'NEUTRAL SENTIMENT', 'POSITIVE SENTIMENT', 'OVERALL SENTIMENT',
            'COLONS', 'SEMI-COLONS', 'COMMAS', 'ELLIPSES', 'SENTENCE LENGTH',
            'PROPER NOUN', 'PUNCTUATION', 'SYMBOL', 'VERB', 'OTHER',
            'SPACE', 'ADJECTIVE', 'ADPOSITION', 'ADVERB', 
            'AUXILLIARY', 'COORDINATING CONJUNCTION', 'DETERMINER',
            'INTERJECTION', 'NOUN', 'NUMERICAL', 'PARTICIPLE',
            'PRONOUN']

POS_hist_dict = dict(zip(POS_abb, POS_desc))

POS_hist_selector_list = []

for k,v in POS_hist_dict.items():
    temp = dict()
    temp['label'] = v
    temp['value'] = k
    POS_hist_selector_list.append(temp)

app = dash.Dash(__name__)

# Putting the startup code for the histograph here to carve it out and keep
# the main block from getting too messy.
data = [
    go.Histogram(
        x=CC_1,
        opacity=.75,
        name='Awesome Films',
        histnorm='percent', 
        xbins={'size':.0004}),
    go.Histogram(
        x=CC_0,
        opacity=.75,
        name='Awful Films',
        histnorm='percent', 
        xbins={'size':.0004})
       ]

layout = go.Layout(
    barmode='overlay',
    legend_orientation='h',
    title='Coordinating Conjunction Frequency')

hist_figure = {'data':data, 'layout':layout}

# Reading in the files for the word cloud images.
image_filename = 'images/good_cloud.png'
encoded_good_cloud = base64.b64encode(
    open(image_filename, 'rb').read()).decode('ascii')

image_filename = 'images/bad_cloud.png'
encoded_bad_cloud = base64.b64encode(
    open(image_filename, 'rb').read()).decode('ascii')

# The main block of code generating the HTML. Using separate file for 
# CSS styling.
app.layout = html.Div(children=[
    html.H1(className='head',
            children=['What\'s a', html.Br(), 'Writer Worth?']
           ),
    html.H2(className='subHead',
            children=["""Screenplay Science and 
            the Value of a Few Good Words"""]
           ),
    html.Div(className='mainBox',
             children=[
        html.Div(className='subBox',
                 children=[
            html.Div(className='thumbBoxLeft',
                     children=[
                html.Img(className='thumbs',
                         src='data:image/png;base64,{}'.format(encoded_good_cloud)
                         ),
                html.Div(className='thumbDesc',
                          children=['The Good']),
                 ]),
            html.Div(className='thumbBoxRight',
                     children=[
                html.Div(className='thumbDesc',
                          children=['The Bad']),
                html.Img(className='thumbs',
                         src='data:image/png;base64,{}'.format(encoded_bad_cloud)
                        )
                 ]),
            # This is where the histogram will go in from the callback.
            html.Div(id='hist', children=[
            dcc.Dropdown(id='hist_selector',
                     options=POS_hist_selector_list,
                     value='NOUN'),
            dcc.Graph(
                id='hist_graph',
                figure=hist_figure)
                ]),               
            ]),
        ]),         
    ])
                        
@app.callback(Output(component_id='hist_graph', 
                     component_property='figure'),
              [Input(component_id='hist_selector',
                     component_property='value')])
def insert_hist(POS_label):
    
    data1 = screenplays_cut[screenplays_cut.good_or_bad == 1]\
    [POS_label]
    
    data0 = screenplays_cut[screenplays_cut.good_or_bad == 0]\
    [POS_label]
    
    std_high_1 = data1.mean() \
        + data1.std()*3
    std_low_1 = data1.mean() \
        - data1.std()*3
    
    plot_info_1 = data1.drop(data1[lambda x: x > std_high_1].index)
    plot_info_1 = plot_info_1.drop(plot_info_1[lambda x: x < std_low_1].index)
    
    std_high_0 = data0.mean() \
        + data0.std()*3
    std_low_0 = data0.mean() \
        - data0.std()*3
    
    plot_info_0 = data0.drop(data0[lambda x: x > std_high_0].index)
    plot_info_0 = plot_info_0.drop(plot_info_0[lambda x: x < std_low_0].index)
    
    bin_size=0
    
    if plot_info_1.mean() > 500:
        bin_size = 100
    if plot_info_1.mean() > 1:
        bin_size = .1
    elif plot_info_1.mean() > .1:
        bin_size = .005
    elif plot_info_1.mean() > .01:
        bin_size = .001
    else:
        bin_size = .0004
#     bin_size = .0004 if plot_info_1.max() < 1 else .1
    
    data = [
    go.Histogram(
        x=plot_info_1,
        opacity=.75,
        name='Awesome Films',
        histnorm='percent', 
        xbins={'size':bin_size}),
    go.Histogram(
        x=plot_info_0,
        opacity=.75,
        name='Awful Films',
        histnorm='percent', 
        xbins={'size':bin_size})
       ]

    layout = go.Layout(
        barmode='overlay',
        legend_orientation='h',
    )

    return {'data':data, 'layout':layout}

#     return ff.create_distplot([plot_info_1, plot_info_0], 
#                               group_labels=['label1','label2'])

if __name__ == '__main__':
    app.run_server(debug=False)
# app