#!/usr/bin/env python
# coding: utf-8

# In[35]:


# import all the required libraries
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
get_ipython().system('pip install nltk')
get_ipython().system('pip install textblob')


# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required resources
nltk.download('stopwords')
nltk.download('vader_lexicon')


# In[36]:


# Read the CSV file into a DataFrame
df = pd.read_csv('sephorareviews.csv')

df.info()


# In[37]:


# printing out all the feature names
print(df.columns.values)


# In[38]:


#display the first five records of the dataframe
df.head()


# In[39]:


print ("Dataset shape: ", df.shape) 


# In[40]:


df.tail()


# In[41]:


#displaying some basic statistical info of the dataframe
df.describe(include=['O'])


# In[42]:


# rename the column to a meaningful name
df.rename(columns={'Unnamed: 0': 'Number'}, inplace=True)
print (df)


# In[43]:


# Display total number of product reviews
total_reviews = len(df)
print("Total number of product reviews:", total_reviews)


# In[44]:


# count the records of the dataframe
df.value_counts()


# # Data Quality Assessment

# In[45]:


# Profiling the data
def profile_data(df):
    profile = df.describe()
    return profile

data_profile = profile_data(df)
print(data_profile)


# In[46]:


# Parse and standardize text data

# Remove spaces in columns and replace them with underscore
df.columns = df.columns.str.replace(" ", "_")

# Set the name of the columns to all lowercase
df.columns = map(str.lower, df.columns)

# Recheck columns
print(df.info())


# In[47]:


# Generalized cleansing function
df.groupby(by = ["number"], dropna = False).count()

# Remove the NaN values from the dataframe
df.dropna(inplace = True)

# Check the NaN again
df.groupby(by = ["number"], dropna= False).count()

# Remove duplicates and check by counting the records
df.drop_duplicates(inplace = True)
df.value_counts()

# Remove unnecessary columns 
df = df.drop(columns=['helpfulness', 'submission_time', 'number'])
df.columns


# In[48]:


# Perform matching based on a keyword in a specific column
def match_by_keyword(df, column, keyword):
    matched_data = df[df[column].str.contains(keyword, case=False)]
    return matched_data

keyword = "good"
matched_data = match_by_keyword(df, 'review_text', keyword)
print("Matched Data:")
print(matched_data)


# In[49]:


# Monitor data for changes or updates
def monitor_data(current_data, previous_data):
    # Implement monitoring logic
    # Example: Compare data statistics or specific fields between current and previous data
    
    # Check if the datasets have the same columns
    if set(current_data.columns) != set(previous_data.columns):
        print("Column mismatch between current and previous data.")
        return
    
    # Compare data statistics
    if current_data.shape != previous_data.shape:
        print("Changes detected in data shape.")
    elif not current_data.equals(previous_data):
        print("Changes detected in data values.")
    else:
        print("No changes detected.")

# Example usage
current_data = pd.read_csv('sephorareviews.csv')
previous_data = pd.read_csv('sephorareviews1.csv')
monitor_data(current_data, previous_data)


# # Problem resolution

# ## Format Checks

# In[50]:


# Check data types of fields
def check_data_types(df):
    data_types = df.dtypes
    return data_types

data_types = check_data_types(df)
print(data_types)


# ## Completeness Check

# In[51]:


# Check for missing values in each column
def check_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

missing_values = check_missing_values(df)
print(missing_values)


# ## Reasonable checks

# In[52]:


# Check if values fall within expected ranges
# there are no invalid ratings here
def check_value_ranges(df):
    # Example: Check if ratings are between 1 and 5
    invalid_ratings = df[(df['rating'] < 1) | (df['rating'] > 5)]
    return invalid_ratings

invalid_ratings = check_value_ranges(df)
print(invalid_ratings)


# ## Limit Checks

# In[53]:


# Check if values exceed or fall below specific limits
def check_limit_values(df):
    # Check if prices exceed $1000
    high_prices = df[df['price_usd'] > 1000]
    return high_prices

high_prices = check_limit_values(df)
print(high_prices)


# ## Missing value handling

# In[54]:


# Impute missing values with mean
def impute_missing_values(df):
    data_filled = df.fillna(df.mean())
    return data_filled

data_filled = impute_missing_values(df)
print(data_filled)


# ## Smooth noisy data

# In[55]:


# Smooth data using a moving average
def smooth_data(df, window=3):
    smoothed_data = df['rating'].rolling(window=window, min_periods=1).mean()
    return smoothed_data

df['smoothed_rating'] = smooth_data(df)
print(df['smoothed_rating'])


# ## Data Analysis and Visualisation

# In[56]:


def textblob_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Return polarity sentiment score
    return sentiment

# Example usage
text = "This product is amazing!"
textblob_score = textblob_sentiment_analysis(text)
print("TextBlob Sentiment Score:", textblob_score)

# the sentiment scores range from -1 to 1, where a score close to 1 indicates a positive sentiment, 
# close to -1 indicates a negative sentiment, and close to 0 indicates a neutral sentiment. 


# In[57]:


def textblob_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Return polarity sentiment score
    return sentiment

texts = ["This product is amazing!", "I'm not happy with the quality.", "The service was excellent!"]

# Perform TextBlob sentiment analysis and sentiment classification for each text
sentiments = [textblob_sentiment_analysis(text) for text in texts]

# Classify sentiments as positive, neutral, or negative based on polarity score
classified_sentiments = []
for sentiment in sentiments:
    if sentiment > 0:
        classified_sentiments.append('Positive')
    elif sentiment < 0:
        classified_sentiments.append('Negative')
    else:
        classified_sentiments.append('Neutral')

# Count the number of texts in each sentiment category
sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
for sentiment in classified_sentiments:
    sentiment_counts[sentiment] += 1

# Create a pie chart to visualize the sentiment distribution
labels = sentiment_counts.keys()
sizes = sentiment_counts.values()
colors = ['green', 'yellow', 'red']
explode = (0.1, 0, 0)  # explode the positive sentiment slice

plt.figure(figsize=(8, 5))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Sentiment Distribution')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[58]:


# bar chart
# Compound score >= 0.05: Positive sentiment
# Compound score <= -0.05: Negative sentiment
# Otherwise: Neutral sentiment

# The code creates a DataFrame to store the texts and their corresponding sentiments. 
# Then, it counts the number of texts in each sentiment category and visualizes the sentiment distribution using a bar plot.

# The updated code will display a bar plot that represents the count of texts in each sentiment category (Positive, Neutral, and Negative).
def vader_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    
    # Classify sentiment based on compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment

# Example usage
texts = ["This product is amazing!", "I'm not happy with the quality.", "The service was excellent!"]

# Perform VADER sentiment analysis and sentiment classification for each text
sentiments = [vader_sentiment_analysis(text) for text in texts]

# Create a DataFrame to store the texts and sentiments
df = pd.DataFrame({'Text': texts, 'Sentiment': sentiments})

# Count the number of texts in each sentiment category
sentiment_counts = df['Sentiment'].value_counts()

# Visualize the sentiment counts
plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()


# In[ ]:





# In[ ]:




