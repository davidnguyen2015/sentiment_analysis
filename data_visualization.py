import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.colors as colors

from wordcloud import WordCloud
from utility import read_data_train, read_data_test

def view_positive_word(df):
    # Visualize Data
    cyberpunk_palette = ["#FF00FF", "#00FF00", "#0000FF"]
    template = "plotly_dark"

    # Word Cloud for Positive Sentiment
    positive_text = ' '.join(df[df['sentiment'] == 'positive']['text'])
    wordcloud = WordCloud(background_color='black', width = 800, height = 400, max_words=200, colormap='Greens').generate(positive_text)
    fig = go.Figure(go.Image(z = np.dstack((wordcloud.to_array(), wordcloud.to_array(), wordcloud.to_array()))))
    fig.update_layout(
        title = 'Word Cloud For Positive Sentiment',
        template = template,
        plot_bgcolor = 'black',
        paper_bgcolor = 'black',
        font_color = cyberpunk_palette[2],
        title_font_color = cyberpunk_palette[2],
        title_font_size = 20,
        margin = dict(t = 80, l = 50, r = 50, b = 50)
    )
    fig.show()

def view_negative_word(df):
    # Visualize Data
    cyberpunk_palette = ["#FF00FF", "#00FF00", "#0000FF"]
    template = "plotly_dark"

    # Word Cloud for Negative Sentiment
    negative_text = ' '.join(df[df['sentiment'] == 'negative']['text'])
    wordcloud = WordCloud(background_color='black', width=800, height=400, max_words=200, colormap='Reds').generate(negative_text)
    fig = go.Figure(go.Image(z=np.dstack((wordcloud.to_array(), wordcloud.to_array(), wordcloud.to_array()))))
    fig.update_layout(
        title="Word Cloud for Negative Sentiment",
        template=template,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color=cyberpunk_palette[2],
        title_font_color=cyberpunk_palette[2],
        title_font_size=20,
        margin=dict(t=80, l=50, r=50, b=50)
    )
    fig.show()

if __name__ == "__main__":
    train = read_data_train()

    view_positive_word(train)