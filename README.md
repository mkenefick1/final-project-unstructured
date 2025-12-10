# final-project-unstructured
This repository contains my final project for Unstructured Data Analytics. The project analyzes fan sentiment from a Notre Dame athletics forum, focusing on discussions surrounding the 2025 Miami–Notre Dame and Stanford–Notre Dame games.

```{python}
# Only loop through single Miami vs. ND page first

import requests
from bs4 import BeautifulSoup
import re
import urllib3
import pandas as pd

# Disable SSL warnings because IrishEnvy triggers certificate errors
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers = {"User-Agent": "Mozilla/5.0"}

url = "https://www.irishenvy.com/threads/aug-31-miami.3051638/"

req_link = requests.get(url, headers=headers, verify=False)

page = BeautifulSoup(req_link.content, "html.parser")

comments = page.select(".message-body")

# stopwords
stopwords = ["the", "and", "is", "to", "of", "a", "an", "in",
"on", "for", "at", "it", "this", "that", "with", "as",
"be", "are", "was", "were", "from", "or", "but", "they",
"if", "them", "their", "there", "about", "just"]

def clean_nd_comments(text, stopwords): 
    text = re.sub(r"^\s*[\w\-.]+\s*(said[:]?|:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[A-Za-z0-9_().\-\s]+?\s*said\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsaid\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Click to expand\.*", "", text)
    text = re.sub(r"\bNA\b", "", text)
    words = text.split()
    text = " ".join([w for w in words if w not in stopwords])
    return text.strip()

links = []

for x in comments:
    not_clean = x.get_text(strip=True)
    cleaned = clean_nd_comments(not_clean, stopwords)
    links.append({
        "thread": url,
        "comment": cleaned
    })

nd_miami = pd.DataFrame(links)
nd_miami.head()
```
Overall, this code block only looped through one page of comments on the Miami vs. Notre Dame game fan forum on irishenvy.com, which took place on August 31st, 2025. Instead of jumping in and looping through all 113 pages right away, the idea was to gather a handful of comments on one page to ensure the code is functioning correctly.

The packages imported at the top of the qmd file (re, requests, BeautifulSoup, urllib3, and pandas) were required to ensure the code runs. The first URL pointed to the first page of Miami comments, which was inspected, and the comments were scraped. 

The stopwords were defined in this code block. Stopwords are unnecessary words that have no impact on this sentiment analysis.

Next, the raw comments were cleaned using regular expressions. They removed usernames like "x said:," "Click to expand," extra whitespace, and the stopwords defined previously.

Then, I cleaned all the first page comments with a for loop, making sure each comment was properly processed and standardized. Finally, I created a dataframe called nd_miami and called the first five rows to ensure the code ran and looked appropriate to move forward.

```{python}
# stopwords
stopwords = ["the", "and", "is", "to", "of", "a", "an", "in",
"on", "for", "at", "it", "this", "that", "with", "as",
"be", "are", "was", "were", "from", "or", "but", "they",
"if", "them", "their", "there", "about", "just"]

# sample
sample = comments[0].get_text(strip=True)
sample

sample_string = str(sample)

# remove any na
sample_no_na = re.sub(r"\bNA\b", "", sample_string)

# remove any stopwords
words = sample_no_na.split()
text = " ".join([w for w in words if w not in stopwords])
sample_no_stopwords = text

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

# URL for miami vs. ND thread
miami_thread_url = "https://www.irishenvy.com/threads/aug-31-miami.3051638/"

# scrape all miami pages
def scrape_miami(thread_url, pages=113): # double checked and 113 pages is the correct number on irishenvy.com
    nd_comments = []

    for page_num in range(1, pages + 1):
        if page_num == 1:
            url = thread_url
        else:
            url = f"{thread_url}page-{page_num}"

        req_miami = requests.get(url, headers=headers, verify=False)
        nd_soup = BeautifulSoup(req_miami.content, "html.parser")

        comments = nd_soup.select(".message-body")
        
        """start at row 1 on page one because the table 
        comes back with the header (row 0) (ATNotre Dame Fighting Irish 
        at Miami (FL) HurricanesHard Rock StadiumMiami, FloridaAugust 31, 20257:30 PM​)) 
        that will mess with my analysis. a staff member posted the game time, day, team, etc. don't need this in my analysis"""
        if page_num == 1:  
            comments = comments[1:]

        for x in comments:
            text = x.get_text(strip=True)
            cleaned = clean_nd_comments(text, stopwords)
            nd_comments.append(cleaned)

    return nd_comments

comments_list = scrape_miami(miami_thread_url)

df = pd.DataFrame(comments_list, columns=["Comment"])
df = df[df['Comment'].str.strip() != ""]

def sentiment(score):
    if score > 0.05: return 'positive'
    elif score < -0.05: return 'negative'
    else: return 'neutral'

df['Sentiment Score'] = df['Comment'].apply(lambda x: vader.polarity_scores(x)['compound'])

df['Sentiment Label'] = df['Sentiment Score'].apply(sentiment)

df.head()

print(df.to_markdown())
```
In this chunk of code, I sampled a stopword list to remove unnecessary words, scraped through the entire Miami vs. ND thread, looping over all 113 pages to collect every comment from this game. Next, I created a dataframe called "df" with all the comments and removed whitespace and blank comments that would impact the analysis.

After, I ran a sentiment analysis by importing SentimentIntensityAnalyzer and defined that a score below -0.05 is negative, above 0.05 is positive, and everything else is neutral. I chose 0.05 and -0.05 as the threshold to categorize sentiments as strongly positive or strongly negative.

I used the head() function to view the first five rows with the sentiments and printed a table in markdown table format for easy access.

```{python}
# bar graph showing distribution of sentiment scores (neutral, positive, negative) - Miami game
import plotly.express as px

plot1 = px.bar(
    df.groupby('Sentiment Label').size().reset_index(name='Count'),
    x='Sentiment Label',
    y='Count',
    title='Distribution of Sentiments – Miami Game',
    color='Sentiment Label',
    text='Count'
)

plot1
```
This bar plot shows the distribution of sentiment scores. Each bar represents the number of positive, negative, and neutral comments. The most comments were in the ___ category for the Miami thread.

```{python}
# Rolling average line graph - Miami game

df['Comment Index'] = range(1, len(df)+1)

df['Rolling_Average'] = df['Sentiment Score'].rolling(window=10).mean()

import plotly.express as px

plot2 = px.line(
    df,
    x='Comment Index',
    y='Rolling_Average',
    title='Rolling Average Sentiment',
)

plot2
```
This line graph shows the rolling average sentiment over the course of the Miami game. There are plenty of spikes and dips in this graph, suggesting the comments were highly reactive to in-game moments.

```{python}
# Now the latest game (ND vs. Stanford) -- stanford first page

import requests
import re
from bs4 import BeautifulSoup
import urllib3

# Disable SSL warnings because IrishEnvy triggers certificate errors
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

headers2 = {"User-Agent": "Mozilla/5.0"}

url2 = "https://www.irishenvy.com/threads/nov-29-stanford.3051649/"


def clean_stanford_comments(text, stopwords):
    text = re.sub(r"^\s*[\w\-.]+\s*(said[:]?|:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[A-Za-z0-9_().\-\s]+?\s*said\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsaid\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Click to expand\.*", "", text)
    text = re.sub(r"\bNA\b", "", text)
    words = text.split()
    text = " ".join([w for w in words if w not in stopwords])
    return text.strip()

req_link_s = requests.get(url2, headers=headers2, verify=False)

page2 = BeautifulSoup(req_link_s.content, "html.parser")

comments2 = page2.select(".message-body")

links2 = []

for x in comments2:
    not_clean = x.get_text(strip=True)
    cleaned = clean_stanford_comments(not_clean, stopwords)
    links2.append({
        "thread": url2,
        "comment": cleaned
    })

nd_stanford = pd.DataFrame(links2)
nd_stanford.head()
```
Overall, this code block only looped through one page of comments on the Stanford vs. Notre Dame game fan forum on irishenvy.com, which took place on November 29th, 2025. Instead of jumping in and looping through all 53 pages right away, the idea was to gather a handful of comments on one page to ensure the code is functioning correctly.

The packages imported at the top of the qmd file (re, requests, BeautifulSoup, urllib3, and pandas) were required to ensure the code runs. The first URL pointed to the first page of Stanford comments, which was inspected, and the comments were scraped. 

Next, the raw comments were cleaned using regular expressions. They removed usernames like "x said:," "Click to expand," extra whitespace, and the stopwords defined previously (stopwords are unnecessary words that have no impact on this sentiment analysis).

Then, I cleaned all the first page comments with a for loop, making sure each comment was properly processed and standardized. Finally, I created a dataframe called nd_stanford and called the first five rows to ensure the code ran and looked appropriate to move forward.

```{python}
# using same Miami stopwords
# stopwords sample2
sample2 = comments2[0].get_text(strip=True)
sample2

sample_string = str(sample2)

# remove any na
sample_no_na = re.sub(r"\bNA\b", "", sample_string)

# remove any stopwords
words = sample_no_na.split()
text = " ".join([w for w in words if w not in stopwords])
sample_no_stopwords = text

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()

# URL for Stanford vs. ND thread
stanford_thread_url = "https://www.irishenvy.com/threads/nov-29-stanford.3051649/"

# scrape all pages
def scrape_stanford(stanford_thread_url, total_pages=53):
    nd_comments2 = []
    for page_num in range(1, total_pages + 1):
        if page_num == 1:
            url2 = stanford_thread_url
        else:
            url2 = f"{stanford_thread_url}page-{page_num}"

        req_link2 = requests.get(url2, headers={"User-Agent" : "Mozilla/5.0"}, verify=False)
        nd_soup2 = BeautifulSoup(req_link2.content, "html.parser")

        comment_blocks = nd_soup2.select(".message-body")

        """start at row 1 on page one because the table 
        comes back with the header (row 0) that will mess with my 
        analysis. a staff member posted the game time, day, 
        team, etc. don't need this in my analysis"""
        if page_num == 1:
            comment_blocks = comment_blocks[1:]

        for x in comment_blocks:
            text = x.get_text(strip=True)
            cleaned = clean_stanford_comments(text, stopwords)
            nd_comments2.append(cleaned)

    return nd_comments2

comments_list2 = scrape_stanford(stanford_thread_url)

df2 = pd.DataFrame(comments_list2, columns=["Comment"])
df2 = df2[df2['Comment'].str.strip() != ""]

df2['Sentiment Score'] = df2['Comment'].apply(lambda x: vader.polarity_scores(x)['compound'])

df2['Sentiment Label'] = df2['Sentiment Score'].apply(sentiment)

df2.head()

print(df2.to_markdown())
```
In this chunk of code, I sampled a stopword list to remove unnecessary words, scraped through the entire Stanford vs. ND thread, looping over all 53 pages to collect every comment from this game. Next, I created a dataframe called "df2" with all the comments and removed whitespace and blank comments that would impact the analysis.

After, I ran a sentiment analysis by importing SentimentIntensityAnalyzer and defined that a score below -0.05 is negative, above 0.05 is positive, and everything else is neutral. I chose 0.05 and -0.05 as the threshold to categorize sentiments as strongly positive or strongly negative.

I used the head() function to view the first five rows with the sentiments and printed a table in markdown table format for easy access.

```{python}
# bar graph showing distribution of sentiment scores (neutral, positive, negative) - Stanford game

plot3 = px.bar(
    df2.groupby('Sentiment Label').size().reset_index(name='Count'),
    x='Sentiment Label',
    y='Count',
    title='Distribution of Sentiments – Stanford Game',
    color='Sentiment Label',
    text='Count'
)

plot3
```


```{python}
# Rolling average line graph - Stanford game
df2['Comment Index'] = range(1, len(df2)+1)

df2['Rolling_Average'] = df2['Sentiment Score'].rolling(window=10).mean()

plot4 = px.line(
    df2,
    x='Comment Index',
    y='Rolling_Average',
    title='Rolling Average Sentiment',
)

plot4
```


```{python}
# concatenate the dfs and add games column to separate
df['Game'] = 'Miami'
df2['Game'] = 'Stanford'

full_comments = pd.concat([df, df2], ignore_index=True)

full_comments

print(full_comments.to_markdown())
```


```{python}
# dfs length check

len(df)
len(df2)

df_length_check = len(df)+len(df2)
length_check = len(full_comments)
length_check
```


```{python}
full_comments['Sentiment Score'] = full_comments['Comment'].apply(lambda x: vader.polarity_scores(x)['compound'])

full_comments['Sentiment Label'] = full_comments['Sentiment Score'].apply(sentiment)

full_comments

print(full_comments.to_markdown())
```


```{python}
miami_avg_sent_score = df['Sentiment Score'].mean()

miami_avg_sent_score
```


```{python}
miami_sent_counts = df['Sentiment Label'].value_counts()

miami_sent_counts

print(miami_sent_counts.to_markdown())
```


```{python}
stan_avg_sent_score = df2['Sentiment Score'].mean()

stan_avg_sent_score
```


```{python}
stan_sent_counts = df2['Sentiment Label'].value_counts()

stan_sent_counts

print(stan_sent_counts.to_markdown())
```


```{python}
full_comments_sent_counts = full_comments['Sentiment Label'].value_counts()

full_comments_sent_counts

print(full_comments_sent_counts.to_markdown())
```


```{python}
full_comments_avg_sent = full_comments['Sentiment Score'].mean()

full_comments_avg_sent
```


```{python}
top_positive = full_comments.sort_values(by='Sentiment Score', ascending=False).head(10)
top_positive[['Game', 'Comment', 'Sentiment Score']]

top_positive

print(top_positive.to_markdown())
```


```{python}
top_negative = full_comments.sort_values(by='Sentiment Score', ascending=True).head(10)
top_negative[['Game', 'Comment', 'Sentiment Score']]

top_negative

print(top_negative.to_markdown())
```


```{python}
# Rolling average line graph - full comments
full_comments['Comment Index'] = range(1, len(full_comments)+1)

full_comments['Rolling_Average'] = full_comments['Sentiment Score'].rolling(window=10).mean()


plot5 = px.line(
    full_comments,
    x='Comment Index',
    y='Rolling_Average',
    title='Rolling Average Sentiment (All Comments)',
)

plot5
```


```{python}
# bar graph showing distribution of sentiment scores (neutral, positive, negative) by game

plot6 = px.bar(
    full_comments.groupby('Sentiment Label').size().reset_index(name='Count'),
    x='Sentiment Label',
    y='Count',
    title='Distribution of Sentiments – Both Games',
    color='Sentiment Label',
    text='Count'
)

plot6
```


```{python}
# bar chart for avg sentiment by miami and stan
import plotly.express as px
import pandas as pd

games = pd.DataFrame({
    "Game": ["Miami", "Stanford"],
    "Average Sentiment": [miami_avg_sent_score, stan_avg_sent_score]
})

plot7 = px.bar(games, x="Game", y="Average Sentiment", color="Game", title="Average Sentiment by Game")

plot7
```


```{python}
# distribution of sentiment scores by game

plot8 = px.histogram(
    full_comments,
    x='Sentiment Score',
    nbins=10,
    title='Distribution of Sentiment Scores by Game',
    color='Game', 
)

plot8
```


```{python}
# sentiment full comments separated by miami and stan

plot9 = px.line(
    full_comments,
    x='Comment Index',
    y='Rolling_Average', 
    color='Game',        
    title='Rolling Avg Sentiment Across Comments by Game'
)

plot9
```
