import streamlit as st
import pandas as pd
import numpy as np
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Plotting
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
import os
os.getcwd()

header = st.beta_container()
dataset = st.beta_container()
topic_modeling = st.beta_container()
vader_sentiment = st.beta_container()
sonar_sentiment = st.beta_container()

with header:
    st.title('Welcome to Suresha HP - Twitter data analytics')
    html_temp = """
        <div style="background-color:#025246 ;padding:10px">
        <h2 style="color:white;text-align:center;">
        Electric cars Twitter data analytics - Python </h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

with dataset:
    st.header('Twitter data of 45000 tweets extracted for 40 Hashtags related to Electric cars')
    st.text('I extracted these tweets from twitter.com')

    df = pd.read_csv('https://raw.githubusercontent.com/hpsuresh12345/Streamlit/main/data/pre-processed.csv')
    st.write(df.head(5))

    st.sidebar.title("Twitter data of 45000 tweets extracted for 40 Hashtags related to Electric cars:")
    st.markdown("I extracted these tweets from twitter.com:")
    st.sidebar.markdown("I extracted these tweets from twitter.com:")
    # frequency count of column Hashtag
    df_hashtags = df['Hashtag'].value_counts()
    labels = df_hashtags.head(25).index.values.tolist()
    freq = df_hashtags.head(25).values.tolist()
    index = np.arange(len(freq))
    st.subheader('25 Top Hashtags')
    plt.figure(figsize=(14, 6))
    plt.bar(index, freq, alpha=0.9, color='green')
    plt.xlabel('Hashtags', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.xticks(index, labels, fontsize=11, rotation=90, fontweight="bold")
    plt.title('Top 25 Hashtags of dataset', fontsize=12, fontweight="bold")
    st.pyplot()
    # Top 25 Most frequent Words
    word_freq = pd.Series(np.concatenate([x.split() for x in df.no_stop_joined])).value_counts()
    word_df = pd.Series.to_frame(word_freq)
    word_df['word'] = list(word_df.index)
    word_df.reset_index(drop=True, inplace=True)
    word_df.columns = ['freq', 'word']
    st.subheader('25 Most frequent Words')
    label = word_df['word'].head(25)
    freq = word_df['freq'].head(25)
    index = np.arange(len(freq))
    print("Unique words:", len(word_df))
    plt.figure(figsize=(12, 9))
    plt.bar(index, freq, alpha=0.8, color='green')
    plt.xlabel('Words', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.xticks(index, label, fontsize=11, rotation=90, fontweight="bold")
    plt.title('Top 25 Words after preprocessing', fontsize=12, fontweight="bold")
    st.pyplot()