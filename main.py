import streamlit as st
import sys
sys.setrecursionlimit(55000)
import pandas as pd
import numpy as np
import nltk
import warnings
import streamlit.components.v1 as components
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Plotting
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
from PIL import Image
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

header = st.beta_container()
dataset = st.beta_container()
topic_modeling = st.beta_container()
vader_sentiment = st.beta_container()
sonar_sentiment =st.beta_container()


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
    df = pd.read_csv(r'C:\Users\hpsur\PycharmProjects\PycharmLearning\Streamlit\data\pre-processed.csv')
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
    plt.figure(figsize=(12,9))
    plt.bar(index, freq, alpha=0.8, color= 'green')
    plt.xlabel('Words', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.xticks(index, label, fontsize=11, rotation=90, fontweight="bold")
    plt.title('Top 25 Words after preprocessing', fontsize=12, fontweight="bold")
    st.pyplot()

# Diaplay wordcloud images

    html_temp = """
            <div style="background-color:#0F75EF ;padding:10px">
            <h2 style="color:white;text-align:center;">
            Word cloud from Tweets </h2>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)

# import Image from pillow to open images
    from PIL import Image

    img = Image.open(r'C:\Users\hpsur\PycharmProjects\PycharmLearning\Streamlit\data\download.png')

# display image using streamlit
# width is used to set the width of an image
    st.image(img, width=600)


with topic_modeling:

    st.header('Topic modeling ')
    html_temp = """
            <div style="background-color:#025122 ;padding:10px">
            <h2 style="color:white;text-align:center;">
            Gensim,Pyldavis - Python </h2>
            </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)

with vader_sentiment:
    st.header('Vader Sentiment ')
    html_temp = """
                <div style="background-color:#025122 ;padding:10px">
                <h2 style="color:white;text-align:center;">
                Vader sentiment analysis - Python </h2>
                </div>
                """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.header('Vader sentiment analysis for the tweets collected')
    # Create an object of Vader Sentiment Analyzer
    df1 = pd.read_csv(r'C:\Users\hpsur\PycharmProjects\PycharmLearning\Streamlit\data\Vader.csv')
    vader_analyzer = SentimentIntensityAnalyzer()

    # Draw Plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(aspect="equal"), dpi=80)

    data = [df1.sentiment.value_counts()[-1], df1.sentiment.value_counts()[0], df1.sentiment.value_counts()[1]]
    categories = ['Negative', 'Neutral', 'Positive']
    explode = [0.05, 0.05, 0.05]


    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}% ({:d} )".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"),
                                      colors=['#e55039', '#3c6382', '#78e08f'],
                                      startangle=140,
                                      explode=explode)

    # Decoration
    ax.legend(wedges, categories, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0.2, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("Number of Tweets by Sentiment", fontsize=12, fontweight="bold")
    st.pyplot()

    # histogram
    labels = ['Negative', 'Neutral', 'Positive']
    freq = [df1.sentiment.value_counts()[-1], df1.sentiment.value_counts()[0], df1.sentiment.value_counts()[1]]
    index = np.arange(len(freq))

    plt.figure(figsize=(8, 6))
    plt.bar(index, freq, alpha=0.8, color='tomato')
    plt.xlabel('Sentiment', fontsize=13)
    plt.ylabel('Number of Tweets', fontsize=13)
    plt.xticks(index, labels, fontsize=11, fontweight="bold")
    plt.title('Number of Tweets per Sentiment', fontsize=12, fontweight="bold")
    plt.ylim(0, len(df1['Tweet Text']))
    st.pyplot()

    # Vader sentiment analyzer
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    #st.write("### Real Time Sentiment Analysis of Electric cars (Vader Sentiment)")

    #user_input = st.text_input("Enter Tweets of Electric cars >>: ")
    #nltk.download("vader_lexicon")
    #s = SentimentIntensityAnalyzer()
    #score = s.polarity_scores(user_input)

    #if score == 0:
        #st.write("Neutral")
    #elif score["neg"] != 0:
        #st.write("# Negative")
    #elif score["pos"] != 0:
        #st.write("# Positive")






