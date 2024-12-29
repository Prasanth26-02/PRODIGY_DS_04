# Step 1: Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Step 2: Load the dataset with assigned column names
column_names = ['ID', 'Platform', 'Sentiment', 'Text']
data = pd.read_csv('/content/ass.csv.csv', header=None, names=column_names)

# Step 3: Text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\W+|\d+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['Cleaned_Text'] = data['Text'].apply(preprocess_text)

# Step 4: Sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

data['Polarity'] = data['Cleaned_Text'].apply(analyze_sentiment)

def categorize_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['Sentiment_Category'] = data['Polarity'].apply(categorize_sentiment)

# Display results
print("Sample Sentiment Analysis Results:")
print(data[['Cleaned_Text', 'Polarity', 'Sentiment_Category']].head())

# Step 5: Save results to a new file
data.to_csv('/content/sentiment_analysis_results.csv', index=False)
print("Sentiment analysis results saved to /content/sentiment_analysis_results.csv")

# Step 6: Visualization - Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment_Category', data=data, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Step 7: Visualization - Word Cloud for Positive Sentiments
positive_text = ' '.join(data[data['Sentiment_Category'] == 'Positive']['Cleaned_Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Positive Sentiments')
plt.show()
