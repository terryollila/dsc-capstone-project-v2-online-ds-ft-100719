
# Abstract

The film industry worldwide does upwards of 50 billion dollars in box office sales, not counting home entertainment revenue, which brings it up closer to 150 billion dollars. Operating within that brick of cash comes with a tremendous amount of risk, with films from major studios sometimes spending a quarter of a billion dollars or more on a single film. Decisions made at smaller studios are no less important to them, as they might be putting their entire livelihoods on the line in the hopes of a hit. And the value of a movie begins with a script.

My premise in creating this project was to ascertain whether a movie's critical rating can be determined to any extent by the text of its screenplay alone. Before attaching a cast and director and crew and all of the other costs associated with creating a cinematic work, having some guidance as to the quality of the script itself can be a benefit in minimizing risk. While an algorithm is no substitute for having a human eye on a screenplay, some level of unbiased machine learning can be leveraged to take a closer look. It is also possible to use this process to vet possible licensing if resources are short and slush piles are large.

As there are many factors going into a movie's rating, such as cast, director, editing, music, costuming, set design and so on, it is not necessarily expected that a movies critical rating can be determined solely by the text of its screenplay. However, there is still much value to be had if any measureable predictability can be found. At the completion of my modeling, I was ultimately able to predict scripts from good movies and bad movies, as rated by metacritic.com, about 65% of the time. Given the other factors in rating, I feel that is a significant enough to create value in the model. Among the various models I tried, some had a better true positive rate, and others had a better false positive rate, so there are some choices there depending on what use cases might be found, such as when it might be more advantageous to find a good movie versus avoiding a bad one.

Recommendations to a given filmmaker would be to use modeling to sort potential screenplays into lists of scripts with higher likelihood of success, using modeling to evaluate scripts in process and step back to consider if it needs more work if the model doesn't like it, and for the screenwriters themselves, to check their scripts against the model and if it comes back with a 'bad' rating, potentially rethink their life choices.

For further research, I would like to create a text ingestion field in the dashboard that allows a user to insert a body of text and have a prediction returned evaluating the content as a screenplay and assigning a good or bad designation. I would create predictability functionality that would allow a user to choose a 'good' or 'bad' setting and have automatically generated text returned back in the style of either a good or bad screenplay. And I would go deeper into the neural networds when modeling, especially toward regression. They showed promise when using them in this project, but there was insufficient time to build them out using pre-made embedding layers and so on.

# Imports


```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd
import numpy as np
import importlib
import requests
import keras
import spacy
import time
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.plotting import plot_confusion_matrix
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence
from bs4 import BeautifulSoup
from tqdm._tqdm_notebook import tqdm_notebook
from tqdm import tqdm
from nltk import word_tokenize
from ast import literal_eval
from importlib import reload

# Importing my own functions file.
import functions as fun
```

    Using TensorFlow backend.



```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```


```python
tqdm_notebook.pandas()
```


```python
nlp = spacy.load('en_core_web_sm')
```

# Obtaining Data

All data for this report will be gathered using web scraping from the following web sites:
    
- metacritic.com for movie rating information.
- rottontomatoes.com for additional movie rating information.
- SpringfieldSpringfield.co.uk for gathering the screenplay texts.

## Rating Data

There were a few attempts at scraping data before finding versions that worked well for my purposes. What follows are the final attempts.

### Sraping metacritic.com

I'm taking the most highly rated and most lowly rated films as listed on this site. These extremes will be used for training my classification models to pridict if random movies will be highly rated or lowly rated films.

#### Great Movies


```python
goods_titles = []

for i in range(0,20):
    # There are 10 pages to flip through of 100 movies each.
    page = requests.get(
        'https://www.metacritic.com/browse/movies/score/metascore/all/filtered?page={}'.format(i),
        headers={'User-Agent': 'Chrome/80.0.3987.116'})
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Now that we've gotten the content from the page, we need to loop through each element.
    for i in range(0,100,1):
        title = soup.find_all('span', class_="title numbered")[i]\
            .next_sibling.next_sibling.contents[1].contents[0]
        goods_titles.append(title)
    
    # We're only pinging 10 times but might as well be safe since it costs like
    # nothing.
    time.sleep(1)
```

Knowing that I won't be able to come up with screenplays for every single movie, I'm taking 2000 great and 2000 terrible films, in the hopes of winding up with at least 1000 of each.


```python
len(goods_titles)
```


```python
goods_titles[:5]
```

#### Terrible Movies


```python
bads_titles = []

for i in range(110, 130):
    # There are 10 pages to flip through of 100 movies each.
    page = requests.get(
        'https://www.metacritic.com/browse/movies/score/metascore/all/filtered?page={}'.format(i),
        headers={'User-Agent': 'Chrome/80.0.3987.116'})
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # Now that we've gotten the content from the page, we need to loop through each element.
    for i in range(0,100,1):
        try:
            title = soup.find_all('span', class_="title numbered")[i]\
            .next_sibling.next_sibling.contents[1].contents[0]
            bads_titles.append(title)
        except:
            pass
    # We're only pinging 10 times but might as well be safe since it costs like
    # nothing.
    time.sleep(1)
```


```python
len(bads_titles)
```


```python
goods_formatted = fun.format_titles(goods_titles)
bads_formatted = fun.format_titles(bads_titles)
```

### Scraping rottentomatoes.com

The rottentomatoes.com information will be used for linear regression. Whereas with metacritic we were using only the best and worst for classification, here I'm using samples from the entire spectrum for regression analysis.


```python
all_rotten_movies = []
rotten_scores = []
for i in range(0, 101):
    page = requests.get("https://www.rottentomatoes.com/browse/"
                        "dvd-streaming-all?minTomato={}&maxTomato={}&services"
                        "=amazon;hbo_go;itunes;netflix_iw;vudu;amazon_prime;"
                        "fandango_now&genres=1;2;4;5;6;8;9;10;11;13;18;14"
                        "&sortBy=release".format(i, i+1))
    soup = BeautifulSoup(page.content, 'html.parser')
    page = soup.get_text()
    comp = re.compile('"\/m\/\w+"')
    movies = comp.findall(page)
    movies_unique = list(set([movie[4:-1] for movie in movies]))
    rotten_scores.extend([i for _ in movies_unique])
    all_rotten_movies.extend(movies_unique)
    print(i)
    print(movies_unique)
    time.sleep(1)
```


```python
# Chopping off the year for those movies that have it.
rotten_movies_noyear = [film[:-5] if film[-4:-2] == '20' else film 
        for film in all_rotten_movies]
```


```python
rotten_form = fun.format_titles(rotten_movies_noyear)
```

## Scraping in the Screenplays

As this will be an analysis centered around natural language processing, my primary data source will be the screenplay text from every movie.

Unfortunately, I was eventually locked out of SpringfieldSpringfield.co.uk, the site where I retrieved the content from, due to too many 'visits'. As I had hit the site upwards of 10k times in the course of a few days, it was probably a fair call.


```python
# Getting the good screenplays along with a list of titles I couldn't find scripts for.
the_good, good_errors = fun.grab_screenplays(goods_formatted)
```


```python
len(good_errors)
```


```python
# Getting the good screenplays along with a list of titles I couldn't find scripts for.
the_bad, bad_errors = fun.grab_screenplays(bads_formatted)
```


```python
len(bad_errors)
```

Putting them both in a DataFrame to be used as data for the rest of the project.


```python
df_good = pd.DataFrame([the_good]).T
df_bad = pd.DataFrame([the_bad]).T
```

Now getting the screenplays for the rottentomatoes titles.


```python
rotten_movies, rotten_errors = fun.grab_screenplays(rotten_form)
```


```python
rotten_df = pd.DataFrame(columns=['titles',
                                  'titles_formatted',
                                  'rotten_scores',
                                  'scripts'])
```


```python
# Getting the titles and scores loaded into the DataFrame.
rotten_df.titles = rotten_form
rotten_df.RottenScores = rotten_scores
```


```python
# Getting things formatted correctly.
rotten_df.scripts = rotten_df.titles_formatted.apply(
    lambda x: rotten_scripts[0][x])
```

Loading all of this data into csv files to be saved in case of setbacks.


```python
df_good.to_csv('../project_resources/df_good_obtain.csv')
df_bad.to_csv('../project_resources/df_bad_obtain.csv')
rotten_df.to_csv('../project_resources/rotten_df_obtain.csv')
```

# Scrubbing Data

Reading back in data that was created in 'obtaining' file.


```python
# I had intended to use files directly from the 'obtain' notebook to be cleaner,
# but I can't get back into the screenplay site, so I'll have to improvise.
# df_good = pd.read_csv('df_good_obtain.csv')
# df_bad = pd.read_csv('df_bad_obtain.csv')
# rotten_df = pd.read_csv('rotten_df_obtain.csv')
```


```python
df_good = pd.read_csv('../project_resources/df_good.csv', index_col=False)
df_bad = pd.read_csv('../project_resources/df_bad.csv', index_col=False)
rotten_df = pd.read_csv('../project_resources/rotten_df.csv', index_col=0)

rotten_df.columns = ['titles', 'titles_formatted', 'rotten_scores', 
                     'scripts', 'all_together_now', 'no_stop', 'just_words']
```

## metacritic

### Setting up DataFrame


```python
# Re-setting columns and index after re-importing.
df_good.columns = ['titles', 'scripts', 'good_or_bad']
df_bad.columns = ['titles', 'scripts', 'good_or_bad']

# Adding labels, combining good and bad, and dropping missing scripts.
df_good['good_or_bad'] = 1
df_bad['good_or_bad'] = 0

screenplays = pd.concat([df_good, df_bad])
screenplays.columns = ['titles', 'scripts', 'good_or_bad']

screenplays.dropna(inplace=True)
```


```python
print('# of good screeenplays: ', len(screenplays[screenplays['good_or_bad'] == 1]))
print('# of bad screenplays: ', len(screenplays[screenplays['good_or_bad'] == 0]))
```

    # of good screeenplays:  1270
    # of bad screenplays:  1514



```python
# converting imported screenplays back to lists from strings
screenplays.scripts = screenplays.scripts.apply(literal_eval)

# Series with a list of lines for each screenplay.
good_to_count = screenplays[screenplays['good_or_bad'] == 1]
bad_to_count = screenplays[screenplays['good_or_bad'] == 0]

# Single string of all good words.
splice_scripts = ''
for script in good_to_count['scripts']:
    splice_scripts += ''.join(script)

all_good_words = ''.join(splice_scripts)

# Single string of all bad words.
splice_scripts = ''
for script in bad_to_count['scripts']:
    splice_scripts += ''.join(script)

all_bad_words = ''.join(splice_scripts)

# Lists of all words lumped together and tokenized
good_data = word_tokenize(all_good_words)
bad_data = word_tokenize(all_bad_words)
```

### Script Metrics

Creating some simple word metrics.


```python
print('good words total: ', len(good_data))
print('bad words total: ', len(bad_data))
print('----'*20)

print('good vocabulary: ', len(set(good_data)))
print('bad vocabulary: ', len(set(bad_data)))
print('----'*20)

print('good % vocab to total: ', round(len(set(good_data)) / len(good_data),4))
print('good % vocab to total: ', round(len(set(bad_data)) / len(bad_data),4))
print('----'*20)

# Total words divided by total number of sripts.
print('Average Good # Words: ', len(good_data) / len(good_to_count))
print('Average Bad # Words: ', len(bad_data) / len(bad_to_count))
print('----'*20)

print('Ave difference by words, good vs bad: ', 
      round(((len(good_data) / len(good_to_count)) \
       - (len(bad_data) / len(bad_to_count))) / (len(bad_data) \
                                                 / len(bad_to_count)),2))
print('----'*20)
```

    good words total:  14020073
    bad words total:  15477699
    --------------------------------------------------------------------------------
    good vocabulary:  172220
    bad vocabulary:  183840
    --------------------------------------------------------------------------------
    good % vocab to total:  0.0123
    good % vocab to total:  0.0119
    --------------------------------------------------------------------------------
    Average Good # Words:  11039.427559055119
    Average Bad # Words:  10223.050858652576
    --------------------------------------------------------------------------------
    Ave difference by words, good vs bad:  0.08
    --------------------------------------------------------------------------------


Counting punctuation and comparing good to bad.


```python
for p in [':', ';', ',', '...', '!']:
    good_p = good_data.count(p) / len(good_data)
    bad_p = bad_data.count(p) / len(bad_data)
    print(f'Good \'{p}\' ratio: ', np.format_float_positional(round(good_p, 6)))
    print(f'Bad \'{p}\' ratio: ', np.format_float_positional(round(bad_p,6)))
    print(f'Good-Bad % for \'{p}\'', round((good_p - bad_p) / bad_p,2))
    print('-----'*10)
```

    Good ':' ratio:  0.001625
    Bad ':' ratio:  0.001428
    Good-Bad % for ':' 0.14
    --------------------------------------------------
    Good ';' ratio:  0.000048
    Bad ';' ratio:  0.000114
    Good-Bad % for ';' -0.58
    --------------------------------------------------
    Good ',' ratio:  0.047181
    Bad ',' ratio:  0.048443
    Good-Bad % for ',' -0.03
    --------------------------------------------------
    Good '...' ratio:  0.011329
    Bad '...' ratio:  0.010441
    Good-Bad % for '...' 0.09
    --------------------------------------------------
    Good '!' ratio:  0.010268
    Bad '!' ratio:  0.014268
    Good-Bad % for '!' -0.28
    --------------------------------------------------


Bad scripts on average use 28% more exclamation marks and 58% more semicolons.

Now creating additional columns for scripts in different formats. Right now, each script line is an element of a list. Breaking those apart into one long string.


```python
splice_scripts = ''
for script in screenplays['scripts']:
    splice_scripts += ''.join(script)

all_words = ''.join(splice_scripts)

temp = []
for script in screenplays['scripts']:
    temp.append(''.join(script))

# This has each script as one long string inside of its cell, 
# as opposed with a list of lines.
screenplays['all_together_now'] = temp

data = word_tokenize(all_words)
```

Looking for # of unique tokens so I know roughly how many to play with when modeling.


```python
len(set(data))
```




    278462



The below function will be used in an apply function remove the stop words for purposes further down. One will keep punctuation and one will remove it.


```python
screenplays['no_stop'] = screenplays['all_together_now']\
    .progress_apply(fun.stop_it, punct=False)

screenplays['just_words'] = screenplays['all_together_now']\
    .progress_apply(fun.stop_it, punct=True)
```


    HBox(children=(IntProgress(value=0, max=2784), HTML(value='')))


    



    HBox(children=(IntProgress(value=0, max=2784), HTML(value='')))


    


## Rottentomatoes

### Setting up DataFrame


```python
rotten_df.dropna(inplace=True)
```


```python
temp = []
for script in rotten_df.scripts:
    temp.append(''.join(script))
```


```python
# This has each script as one long string inside of its cell, 
# as opposed with a list of lines.
rotten_df['all_together_now'] = temp
```

Using this function again for rotten_df this time to remove stop words for one columns of scripts and remove stop words plus punctuation for another.


```python
rotten_df['no_stop'] = rotten_df['all_together_now']\
    .progress_apply(fun.stop_it, punct=False)

rotten_df['just_words'] = rotten_df['all_together_now']\
    .progress_apply(fun.stop_it, punct=True)
```


    HBox(children=(IntProgress(value=0, max=1536), HTML(value='')))


    



    HBox(children=(IntProgress(value=0, max=1536), HTML(value='')))


    



```python
# Need to rename the period since it causes problems later on.
rotten_df.rename(columns={'.':'PER'}, inplace=True)
```


```python
# rotten_df.dropna(inplace=True)
# screenplays.dropna(inplace=True)

rotten_df = rotten_df.drop_duplicates(subset=['titles']).copy()
screenplays = screenplays.drop_duplicates(['titles']).copy()

screenplays.to_csv('../project_resources/screenplays_scrub.csv')
rotten_df.to_csv('../project_resources/rotten_df_scrub.csv')
```

# Exploratory Data Analysis

## Local Functions


```python
# Counting punctuation and parts of speech
def count_punct(text, punk):
    return text.count(punk) / len(text.split())

def count_pos(x):
    return x.count_by(spacy.attrs.POS)

def count_tag(x):
    return x.count_by(spacy.attrs.TAG)

# Determining the average sentence length for a given script.
def sent_len(str):
    doc = nlp(str)
    count = 0
    
    for sent in doc.sents:
        count += 1
    
    return len(str.split()) / count

# These two find the acutal POS name for the code number supplied, for
# creating the column names.
def POS_reverse_lookup_n_ratio(x, code):
    try:
        return x.POS_counts[code] / x.word_count
    except:
        pass

def TAG_reverse_lookup_n_ratio(x, code):
    try:
        return x.TAG_counts[code] / x.word_count
    except:
        return 0
```

## Bringing in the data

Reading back in data we scubbed in 'scrubbing'. 


```python
screenplays = pd.read_csv('../project_resources/screenplays_scrub.csv', index_col=0)
rotten_df = pd.read_csv('../project_resources/rotten_df_scrub.csv', index_col=0)

screenplays.dropna(inplace=True)
```


```python
# Adding column showing if a title is above or below a score of 50.
rotten_df['good_or_bad'] = rotten_df.rotten_scores.apply(
    lambda x: 1 if x >=50 else 0)

# Re-ordering columns to line up the two data sources for later use.
screen_cols = list(screenplays.columns)
screen_cols.remove('good_or_bad')
screen_cols.append('good_or_bad')
screenplays = screenplays[screen_cols].copy()

rotten_cols = list(rotten_df.columns)
rotten_cols.remove('rotten_scores')
rotten_cols.append('rotten_scores')
rotten_df = rotten_df[rotten_cols].copy()
rotten_df.drop('titles', axis=1, inplace=True)
```


```python
# rotten_df = rotten_df[sort_cols]
rotten_df.rename(columns={'titles_formatted':'titles'}, inplace=True)
```

## Latent Derichlet Allocation

Finding the categories inherent in the screenplays by analyzing the words and grouping the screenplays.


```python
combined_df = pd.concat([rotten_df[['titles', 'just_words']],
                         screenplays[['titles', 'just_words']]],
                         ignore_index=True)

combined_df.drop_duplicates(subset='titles', inplace=True)
```


```python
cv = CountVectorizer(min_df=.1, max_df=.6)
victor = cv.fit_transform(combined_df.just_words)
LDA = LatentDirichletAllocation(n_components=10, random_state=42)
combined_df['category'] = LDA.fit_transform(victor).argmax(axis=1)
```

Showing the category word groupings, then creating materiels to be used later.


```python
# This will house the dictionaries for using in the dashboard.
cat_word_dicts = []

# Priting out the word gropuings and filling in the cat_word_dicts list.
for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR CATEGORY #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    cat_word_dicts.append({index:[cv.get_feature_names()[i]\
                                  for i in topic.argsort()[-15:]]})
#     cat_word_dicts.append([cv.get_feature_names()[i]\
#                           for i in topic.argsort()[-15:]])

cat_word_df = pd.DataFrame()

# Creating a dataframe for export to .csv for later use.
for i in range(len(cat_word_dicts)):
    cat_word_df[i] = cat_word_dicts[i][i]

# Creating category mapping for all dataframes to indicate how I see 
# The groupings. This part is just based on my looking them over.
category_map = {0: 'Dark & Political', 1: 'Sports, Comedy, Silly Horror', 
                2: 'Conflict', 3: 'Holiday, Films I Haven\'t Seen',
                4: 'Light-Hearted', 5: 'Unusual Language or Slang', 
                6: 'Violence & Gangster', 7: 'Romance & Light Drama', 
                8: 'Life Stories', 9: 'Straight Up Horror'}
combined_df['category_label'] = combined_df.category.map(category_map)
```

    THE TOP 15 WORDS FOR CATEGORY #0
    ['government', 'fight', 'human', 'war', 'law', 'body', 'children', 'power', 'court', 'country', 'state', 'president', 'death', 'police', 'dr']
    
    
    THE TOP 15 WORDS FOR CATEGORY #1
    ['buddy', 'wow', 'team', 'cool', 'uh', 'ball', 'aint', 'ya', 'ah', 'em', 'dog', 'ha', 'game', 'whoa', 'ok']
    
    
    THE TOP 15 WORDS FOR CATEGORY #2
    ['air', 'safe', 'jim', 'ship', 'war', 'jack', 'officer', 'clear', 'shot', 'shoot', 'fire', 'police', 'john', 'captain', 'gun']
    
    
    THE TOP 15 WORDS FOR CATEGORY #3
    ['ooh', 'whoa', 'ah', 'christmas', 'cool', 'wow', 'bye', 'daddy', 'charlie', 'sam', 'honey', 'jack', 'um', 'uh', 'mom']
    
    
    THE TOP 15 WORDS FOR CATEGORY #4
    ['uh', 'grunts', 'radio', 'crowd', 'chatter', 'cheering', 'speaking', 'indistinct', 'continues', 'laughs', 'playing', 'laughing', 'sighs', 'chuckles', 'music']
    
    
    THE TOP 15 WORDS FOR CATEGORY #5
    ['aint', 'poor', 'sister', 'darling', 'war', 'brother', 'child', 'death', 'lady', 'children', 'king', 'lord', 'dear', 'shall', 'mrs']
    
    
    THE TOP 15 WORDS FOR CATEGORY #6
    ['gun', 'brother', 'goddamn', 'motherfucker', 'dude', 'fucked', 'em', 'yo', 'jesus', 'bitch', 'ass', 'fuckin', 'aint', 'fucking', 'fuck']
    
    
    THE TOP 15 WORDS FOR CATEGORY #7
    ['buy', 'women', 'parents', 'york', 'bye', 'perfect', 'dinner', 'book', 'party', 'honey', 'movie', 'girls', 'sex', 'married', 'mom']
    
    
    THE TOP 15 WORDS FOR CATEGORY #8
    ['white', 'women', 'ii', 'sort', 'mmm', 'mm', 'wow', 'alright', 'mmhmm', 'john', 'bob', 'hmm', 'ah', 'um', 'uh']
    
    
    THE TOP 15 WORDS FOR CATEGORY #9
    ['groaning', 'breathing', 'chuckles', 'continues', 'music', 'groans', 'screams', 'panting', 'mary', 'sighs', 'michael', 'grunting', 'grunts', 'gasps', 'screaming']
    
    


I'm going to need the combined dataframes with LDA categories for the dashboard later on.


```python
screenplays = pd.merge(screenplays, combined_df[['titles','category', 'category_label']],
                how='left', on='titles')

rotten_df = pd.merge(rotten_df, combined_df[['titles','category', 'category_label']],
                how='left', on='titles')

cat_word_df.to_csv('../project_resources/cat_word_df.csv', header=True)

combined_df.to_csv('../project_resources/combined_df.csv')
```

## Metacritic

### Feature Engineering

Will be creating quite a few new columns filled with script attributes such as word counts, sentiment, and parts of speech.

Putting in some counts.


```python
screenplays['word_count'] = screenplays.all_together_now.apply(
    lambda x: len(x.split()))
screenplays['unique_words'] = screenplays.no_stop.apply(
    lambda x: len(x.split()))
```

Creating columns for sentiment scoring: positive, negative, neutral, and compound.


```python
sid = SentimentIntensityAnalyzer()

screenplays['sentiment_scores'] = screenplays.no_stop.progress_apply(
    lambda x: sid.polarity_scores(x))

screenplays['sentiment_negative'] = screenplays.sentiment_scores.apply(
    lambda x: x['neg'])
screenplays['sentiment_neutral'] = screenplays.sentiment_scores.apply(
    lambda x: x['neu'])
screenplays['sentiment_positive'] = screenplays.sentiment_scores.apply(
    lambda x: x['pos'])
screenplays['sentiment_compound'] = screenplays.sentiment_scores.apply(
    lambda x: x['compound'])
```


    HBox(children=(IntProgress(value=0, max=2765), HTML(value='')))


    


Removing anything with fewer words words so that short scripts don't throw off the modeling.


```python
screenplays_cut = screenplays.copy().drop(
    index=(screenplays[screenplays['word_count'] < 1000].index), axis=0)
```

More columns for punctuation ratios.


```python
screenplays_cut['colon_ratios'] = screenplays_cut.no_stop.apply(
    lambda x: count_punct(x, ':'))
screenplays_cut['semi_ratios'] = screenplays_cut.no_stop.apply(
    lambda x: count_punct(x, ';'))
screenplays_cut['comma_ratios'] = screenplays_cut.no_stop.apply(
    lambda x: count_punct(x, ','))
screenplays_cut['ellipsis_ratios'] = screenplays_cut.no_stop.apply(
    lambda x: count_punct(x, '...'))
```

And sentence length.


```python
screenplays_cut['sentence_length'] = screenplays_cut.no_stop.progress_apply(
    sent_len)
```


    HBox(children=(IntProgress(value=0, max=2745), HTML(value='')))


    


### Parts of Speech

Breaking out both coarse and fine grained approaches to parts of speech, to create a ratio of how many of each is used in each screenplay.


```python
screenplays_cut['nlp'] = screenplays_cut.all_together_now.progress_apply(nlp)
screenplays_cut.to_csv('../project_resources/screenplays_cut.csv')
```


    HBox(children=(IntProgress(value=0, max=2745), HTML(value='')))


    



```python
# Applying the 'progress_apply' function to each row to get the actual words.
screenplays_cut['POS_counts'] = screenplays_cut.nlp.progress_apply(count_pos)
screenplays_cut['TAG_counts'] = screenplays_cut.nlp.progress_apply(count_tag)

# POS_codes and TAG codes will be used to identify unique codes 
# used throughout vocabulary.
POS_codes = set()

for i, item in enumerate(screenplays_cut.POS_counts):
    POS_codes.update(item.keys())

TAG_codes = set()

for i, item in enumerate(screenplays_cut.TAG_counts):
    TAG_codes.update(item.keys())
```


    HBox(children=(IntProgress(value=0, max=2745), HTML(value='')))


    



    HBox(children=(IntProgress(value=0, max=2745), HTML(value='')))


    



```python
all_scripts = set()

for i, script in enumerate(screenplays_cut.just_words):
    all_scripts.update(script.split())

all_scripts = [script for script in all_scripts]

all_scripts = ' '.join(all_scripts)

all_scripts = nlp(all_scripts[:30])
```

Adding POS columns.


```python
# TAG_lookup will be used for adding POS word to column name.
TAG_lookup = {}
for code in TAG_codes:
    key = code
    value = all_scripts.vocab[code].text
    TAG_lookup[key] = value

POS_lookup = {}
for code in POS_codes:
    key = code
    value = all_scripts.vocab[code].text
    POS_lookup[key] = value

# Adding POS column and adding POS name as column name.
for code, abb in POS_lookup.items():
    screenplays_cut[abb] = screenplays_cut.apply(
        POS_reverse_lookup_n_ratio, args=[code], axis=1)

# Adding TAG column and adding TAG name as column name.
for code, abb in TAG_lookup.items():
    screenplays_cut[abb] = screenplays_cut.apply(
        TAG_reverse_lookup_n_ratio, args=[code], axis=1)
```


```python
screenplays_cut = screenplays_cut.fillna(0)

screenplays_cut.to_csv('../project_resources/screenplays_cut.csv')
```

## Rottentomatoes

### Feature Engineering

As with metacritic, will be creating quite a few new columns filled with script attributes such as word counts, sentiment, and parts of speech.

Putting in some counts.


```python
rotten_df['word_count'] = rotten_df.all_together_now.apply(
    lambda x: len(x.split()))
rotten_df['unique_words'] = rotten_df.no_stop.apply(
    lambda x: len(x.split()))
```


```python
rotten_df_cut = rotten_df.copy().drop(
    index=(rotten_df[rotten_df['word_count'] < 1000].index), axis=0)
```

Creating columns for sentiment scoring: positive, negative, neutral, and compound.


```python
sid = SentimentIntensityAnalyzer()

rotten_df_cut['sentiment_scores'] = rotten_df.no_stop.progress_apply(
    lambda x: sid.polarity_scores(x))

rotten_df_cut['sentiment_negative'] = rotten_df_cut.sentiment_scores.apply(
    lambda x: x['neg'])
rotten_df_cut['sentiment_neutral'] = rotten_df_cut.sentiment_scores.apply(
    lambda x: x['neu'])
rotten_df_cut['sentiment_positive'] = rotten_df_cut.sentiment_scores.apply(
    lambda x: x['pos'])
rotten_df_cut['sentiment_compound'] = rotten_df_cut.sentiment_scores.apply(
    lambda x: x['compound'])
```


    HBox(children=(IntProgress(value=0, max=1536), HTML(value='')))


    


Removing anything with fewer words words so that short scripts don't throw off the modeling.

More columns for punctuation ratios.


```python
rotten_df_cut['colon_ratios'] = rotten_df_cut.no_stop.apply(
    lambda x: count_punct(x, ':'))
rotten_df_cut['semi_ratios'] = rotten_df_cut.no_stop.apply(
    lambda x: count_punct(x, ';'))
rotten_df_cut['comma_ratios'] = rotten_df_cut.no_stop.apply(
    lambda x: count_punct(x, ','))
rotten_df_cut['ellipsis_ratios'] = rotten_df_cut.no_stop.apply(
    lambda x: count_punct(x, '...'))
```

And sentence length.


```python
rotten_df_cut['sentence_length'] = rotten_df_cut.no_stop.progress_apply(
    sent_len)
```


    HBox(children=(IntProgress(value=0, max=1530), HTML(value='')))


    


### Parts of Speech

Breaking out both coarse and fine grained approaches to parts of speech, to create a ratio of how many of each is used in each screenplay.


```python
rotten_df_cut['nlp'] = rotten_df_cut.all_together_now.progress_apply(nlp)
```


    HBox(children=(IntProgress(value=0, max=1530), HTML(value='')))


    



```python
rotten_df_cut.to_csv('../project_resources/rotten_df_cut.csv')
```


```python
# rotten_df_cut = pd.read_csv('../project_resources/rotten_df_cut.csv')
```


```python
# Applying the 'progress_apply' function to each row to get the actual words.
rotten_df_cut['POS_counts'] = rotten_df_cut.nlp.progress_apply(count_pos)

rotten_df_cut['TAG_counts'] = rotten_df_cut.nlp.progress_apply(count_tag)

# POS_codes and TAG codes will be used to identify unique codes 
# used throughout vocabulary.
POS_codes = set()

for i, item in enumerate(rotten_df_cut.POS_counts):
    POS_codes.update(item.keys())

TAG_codes = set()

for i, item in enumerate(rotten_df_cut.TAG_counts):
    TAG_codes.update(item.keys())
```


    HBox(children=(IntProgress(value=0, max=1530), HTML(value='')))


    



    HBox(children=(IntProgress(value=0, max=1530), HTML(value='')))


    


Adding POS columns.


```python
# TAG_lookup will be used for adding POS word to column name.
TAG_lookup = {}
for code in TAG_codes:
    key = code
    value = all_scripts.vocab[code].text
    TAG_lookup[key] = value
    
POS_lookup = {}
for code in POS_codes:
    key = code
    value = all_scripts.vocab[code].text
    POS_lookup[key] = value

# Adding POS column and adding POS name as column name.
for code, abb in POS_lookup.items():
    rotten_df_cut[abb] = rotten_df_cut.apply(
        POS_reverse_lookup_n_ratio, args=[code], axis=1)

# Adding TAG column and adding TAG name as column name.
for code, abb in TAG_lookup.items():
    rotten_df_cut[abb] = rotten_df_cut.apply(
        TAG_reverse_lookup_n_ratio, args=[code], axis=1)
```


```python
rotten_df_cut = rotten_df_cut.fillna(0)
```


```python
rotten_df_cut.to_csv('../project_resources/rotten_df_cut.csv')
```

Note that in the original eda.ipynb file, I have an extensive set of scatter matrices, histograms, and correlation matrices at this point in the process. However, they took up too much memory to be efficiently placed in this document, but they can be referred to in eda.ipynb as needed.

# Modeling

## Bring in the data

Data previously saved in .csv format from eda.ipynb.


```python
rotten_df_cut = pd.read_csv('../project_resources/rotten_df_cut.csv', index_col=0)
screenplays_cut = pd.read_csv('../project_resources/screenplays_cut.csv', index_col=0)
```

Shuffling columns  for ease of use when modeling.


```python
screen_cols = list(screenplays_cut.columns)
screen_cols.remove('good_or_bad')
screen_cols.append('good_or_bad')
screenplays_model = screenplays_cut[screen_cols].copy()

rotten_cols = list(rotten_df_cut.columns)
rotten_cols.remove('good_or_bad')
rotten_cols.append('good_or_bad')
rotten_cols.remove('rotten_scores')
rotten_cols.append('rotten_scores')
rotten_model = rotten_df_cut[rotten_cols].copy()
```


```python
# sort_cols = list(rotten_model.columns[:38])
# sort_cols.extend(sorted(list(rotten_model.columns[38:-2])))
# sort_cols.extend(list(rotten_model.columns[-2:]))
```

## TFIDF Vectorization

My first set of models will employ TFIDF Vectorization using a variety of classifiers.m

### Setup

For the modeling, I'm going to use the extreme ratings I got from metacritic.com for training the models, then use the rottentomatoes data, which is uniformly distributed from bad movies to good movies, to test the modeling.


```python
X_train = screenplays_cut.no_stop
X_test = rotten_df_cut.no_stop

y_train = screenplays_cut.good_or_bad
y_test = rotten_df_cut.good_or_bad
```

### Linear SVC


```python
fun.hybrid_classifiers(X_train, X_test, y_train, y_test, LinearSVC())
```

                  precision    recall  f1-score   support
    
               0       0.64      0.77      0.70       811
               1       0.67      0.52      0.58       719
    
        accuracy                           0.65      1530
       macro avg       0.65      0.64      0.64      1530
    weighted avg       0.65      0.65      0.65      1530
    



![png](main_files/main_146_1.png)


    roc_auc score:  0.7007249073500837



![png](main_files/main_146_3.png)





    array([0, 0, 1, ..., 1, 0, 0])



### SVC


```python
fun.hybrid_classifiers(X_train, X_test, y_train, y_test, SVC())
```

                  precision    recall  f1-score   support
    
               0       0.65      0.77      0.71       811
               1       0.67      0.54      0.60       719
    
        accuracy                           0.66      1530
       macro avg       0.66      0.65      0.65      1530
    weighted avg       0.66      0.66      0.66      1530
    



![png](main_files/main_148_1.png)


    roc_auc score:  0.714379301297013



![png](main_files/main_148_3.png)





    array([0, 0, 1, ..., 1, 1, 0])



### XGBoost Classifier


```python
fun.hybrid_classifiers(X_train, X_test, y_train, y_test, 
                   XGBClassifier(max_depth=8,
                                    criterion='entropy',
                                    min_samples_split=14,
                                    min_samples_leaf=1,
                                    max_features=160))
```

                  precision    recall  f1-score   support
    
               0       0.65      0.80      0.72       811
               1       0.69      0.52      0.59       719
    
        accuracy                           0.67      1530
       macro avg       0.67      0.66      0.66      1530
    weighted avg       0.67      0.67      0.66      1530
    



![png](main_files/main_150_1.png)





    array([0, 0, 0, ..., 1, 1, 0])



### Neural Network

Because the neural network works so differently from the other classifiers, I haven't used a function, and the main code is all here.

Starting to put the data together.


```python
X_train = pd.DataFrame(X_train)

# Undersampling to get things even just for ease of use.
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(pd.DataFrame(X_train), y_train)

X_resampled = X_resampled.iloc[:, 0]
```

Create the vector from the full batch of scripts.


```python
tfidf = TfidfVectorizer(max_features=5000, max_df=.95, min_df=.1, 
                        ngram_range=(1,2))
X_t_train = tfidf.fit_transform(X_resampled)
X_t_test = tfidf.transform(X_test)
```

Getting the data into the format it will need to be in for the network.


```python
X_t_train = pd.SparseDataFrame(X_t_train, columns=tfidf.get_feature_names(),
                           default_fill_value=0)
X_t_test = pd.SparseDataFrame(X_t_test, columns=tfidf.get_feature_names(),
                           default_fill_value=0)

X_t_num = np.array(X_t_train)
X_t_test_num = np.array(X_t_test)

y_t_num = np.array(y_resampled)
y_t_test_num = np.array(y_test)

layer_input = X_t_train.shape[1]
```

I played manually with a lot of different layers, neurons, drop layers, and regularization. Simple seemed to work best.


```python
model = Sequential()

model.add(Dense(layer_input, input_dim=layer_input, activation='relu'))
model.add(Dense(50, input_dim=layer_input, activation='relu'))
model.add(Dense(50, input_dim=50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```


```python
model.compile(loss='mean_squared_error',
              optimizer='adam', 
              metrics=['accuracy'])

model.fit(X_t_num, y_t_num, epochs=4, batch_size=50, validation_split=0.2)
```

    Train on 2006 samples, validate on 502 samples
    Epoch 1/4
    2006/2006 [==============================] - 7s 4ms/step - loss: 0.1997 - accuracy: 0.7009 - val_loss: 0.3002 - val_accuracy: 0.5378
    Epoch 2/4
    2006/2006 [==============================] - 6s 3ms/step - loss: 0.1248 - accuracy: 0.8295 - val_loss: 0.2312 - val_accuracy: 0.6594
    Epoch 3/4
    2006/2006 [==============================] - 7s 3ms/step - loss: 0.0626 - accuracy: 0.9212 - val_loss: 0.1981 - val_accuracy: 0.7291
    Epoch 4/4
    2006/2006 [==============================] - 6s 3ms/step - loss: 0.0321 - accuracy: 0.9636 - val_loss: 0.2210 - val_accuracy: 0.7171





    <keras.callbacks.callbacks.History at 0x65d4ef6d8>




```python
# y_t_test = y_test
```


```python
model.evaluate(X_t_test_num, y_t_test_num, verbose=1)
```

    1530/1530 [==============================] - 0s 300us/step





    [0.29038501397457, 0.6392157077789307]




```python
y_pred = model.predict_classes(X_t_test_num)

print(classification_report(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
                  show_normed=True, cmap=plt.cm.Reds);
plt.show();
```

                  precision    recall  f1-score   support
    
               0       0.62      0.81      0.70       811
               1       0.67      0.45      0.54       719
    
        accuracy                           0.64      1530
       macro avg       0.65      0.63      0.62      1530
    weighted avg       0.65      0.64      0.63      1530
    



![png](main_files/main_164_1.png)


Overall, a pretty positive model.

### TFIDF Summary

The models here were strong, overall. Most of them had accuracy of around 65%, with poor true positive rate of around 50, which means they were quite good at predicting which scripts were bad but not great at predicting which scripts were good. The decision tree had good true positive rate at 61%, but the accuracy wasn't as good at 58%. XG Boost had the best accuracy at 66, but the true positive rate of precisely 50%, with true negative rate 80%.

## Script Attributes

### Setup

For this set of modeling, I'm using only the script attributes from the dataframe, so basically word counts, punctuation ratios, sentence length, sentiment, and parts of speech.


```python
screen_dummies = pd.get_dummies(screenplays_model['category'])
rotten_dummies = pd.get_dummies(rotten_model['category'])
```


```python
columns = list(screenplays_model.columns[7:9])
columns.extend(list(screenplays_model.columns[10:19]))
start = list(screenplays_model.columns).index('PROPN')
columns.extend(list(screenplays_model.columns[start:-1]))
    
X = screenplays_model[columns]
X_train = pd.merge(screenplays_model[columns], screen_dummies, left_index=True, 
             right_index=True)
X_test = pd.merge(rotten_model[columns], rotten_dummies, left_index=True, 
             right_index=True)

y_train = screenplays_model.good_or_bad
y_test = rotten_model.good_or_bad
```

### XG Boost Classifier


```python
fun.hybrid_classifiers(X_train, X_test, y_train, y_test, 
                   classifier=XGBClassifier(),use_tfidf=False)
```

                  precision    recall  f1-score   support
    
               0       0.59      0.43      0.50       811
               1       0.51      0.66      0.57       719
    
        accuracy                           0.54      1530
       macro avg       0.55      0.54      0.54      1530
    weighted avg       0.55      0.54      0.53      1530
    



![png](main_files/main_174_1.png)





    array([0, 0, 1, ..., 1, 1, 0])



### Random Forest Classifier


```python
fun.hybrid_classifiers(X_train, X_test, y_train, y_test, 
                   classifier=RandomForestClassifier(random_state=42,
                                                    max_depth=8,
                                                    criterion='entropy',
                                                    min_samples_split=14,
                                                    min_samples_leaf=1,
#                                                     max_features=10),
                                                    ),
                  use_tfidf=False)
```

                  precision    recall  f1-score   support
    
               0       0.60      0.42      0.49       811
               1       0.51      0.69      0.59       719
    
        accuracy                           0.55      1530
       macro avg       0.56      0.55      0.54      1530
    weighted avg       0.56      0.55      0.54      1530
    



![png](main_files/main_176_1.png)





    array([0, 1, 1, ..., 0, 1, 0])



### Scripts Attributes Summary

This set was much lower than the TFIDF, with scores that were only a little above random chance. The neural network simply classified everything as positive. I'm sure I could have done something to improve this batch given time, but for now, as it was so much lower than the TFIDF set, I chose to let it go.

## Combined TDIF & Attributes

### Setup

For this more experimental bunch, I thought I would try merging the TFIDF vecctorization matrix with the attributes from the original dataframe, wondering if more information might be better, and how the engineered features might play against the word matrix.


```python
screen_dummies = pd.get_dummies(screenplays_model['category'])
rotten_dummies = pd.get_dummies(rotten_model['category'])
```


```python
columns = list(screenplays_model.columns[7:9])
columns.extend(list(screenplays_model.columns[10:19]))
start = list(screenplays_model.columns).index('PROPN')
columns.extend(list(screenplays_model.columns[start:-1]))

X_train = screenplays_model[columns].copy()
X_test = rotten_model[columns].copy()

y_train = screenplays_model.good_or_bad.copy()
y_test = rotten_model.good_or_bad.copy()
```


```python
X2_train = screenplays_model.no_stop
X2_test = rotten_model.no_stop
y_train = screenplays_model.good_or_bad
y_test = rotten_model.good_or_bad
```

### Linear Support Vector Classifier


```python
# temp = pd.DataFrame(X2_train, columns=['temp'])
# temp[temp['temp'].isna() == True].index
```


```python
# screenplays_model.iloc[[40, 69, 101, 106, 147, 175, 264, 303, 343, 371, 392,
#              464, 656, 811, 963, 1099, 2024, 2044, 2066, 2265, 2554, 2600]]
```


```python
fun.hybrid_classifier_combo(X_train, X_test, X2_train, X2_test,
                        y_train, y_test, LinearSVC(C=.6))
```

    //anaconda3/lib/python3.7/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning:
    
    Liblinear failed to converge, increase the number of iterations.
    


                  precision    recall  f1-score   support
    
               0       0.25      0.00      0.00       811
               1       0.47      1.00      0.64       719
    
        accuracy                           0.47      1530
       macro avg       0.36      0.50      0.32      1530
    weighted avg       0.35      0.47      0.30      1530
    
    Accuracy:  0.46862745098039216



![png](main_files/main_188_2.png)


    roc_auc score:  0.5150820858535883



![png](main_files/main_188_4.png)


### Logistic Regression Classifier


```python
fun.hybrid_classifier_combo(X_train, X_test, X2_train, X2_test,
                        y_train, y_test, LogisticRegression(C=10))
```

                  precision    recall  f1-score   support
    
               0       0.53      0.99      0.69       811
               1       0.61      0.02      0.04       719
    
        accuracy                           0.53      1530
       macro avg       0.57      0.50      0.36      1530
    weighted avg       0.57      0.53      0.38      1530
    
    Accuracy:  0.5333333333333333



![png](main_files/main_190_1.png)


    roc_auc score:  0.498411103241418



![png](main_files/main_190_3.png)


### Random Forest Classifier


```python
fun.hybrid_classifier_combo(X_train, X_test, X2_train, X2_test,
                        y_train, y_test, 
                        RandomForestClassifier(random_state=42, 
                                               n_jobs=-1),
                        feature_importance=False)
```

                  precision    recall  f1-score   support
    
               0       0.65      0.76      0.70       811
               1       0.67      0.55      0.60       719
    
        accuracy                           0.66      1530
       macro avg       0.66      0.65      0.65      1530
    weighted avg       0.66      0.66      0.65      1530
    
    Accuracy:  0.6588235294117647



![png](main_files/main_192_1.png)


## Stacked Modeling

This is one of my top models from the TFIDF modeling section. I'll use it as the first phase of a two-part modeling scenario.

First, I'm going to send in the movies from metacritic to train the first model, using the rottentomatoes data for testing. Then I'll marry the test predictions to the rottentomatoes (test) data, and send it through the second model, where it will get split into a proper train-test split.


```python
X_train = screenplays_cut.no_stop
X_test = rotten_df_cut.no_stop

y_train = screenplays_cut.good_or_bad
y_test = rotten_df_cut.good_or_bad
```


```python
model_1_predictions = fun.hybrid_classifiers(X_train, X_test, y_train, y_test, SVC())
```

                  precision    recall  f1-score   support
    
               0       0.65      0.77      0.71       811
               1       0.67      0.54      0.60       719
    
        accuracy                           0.66      1530
       macro avg       0.66      0.65      0.65      1530
    weighted avg       0.66      0.66      0.66      1530
    



![png](main_files/main_196_1.png)


    roc_auc score:  0.714379301297013



![png](main_files/main_196_3.png)



```python
# X = screenplays_model.no_stop
# y = screenplays_model.good_or_bad

# model_1_predictions = fun.stacked_classifier(X, y, SVC())
```

Below I'm getting the dummies and putting them all in one dataframe to be joined later.


```python
screen_dummies = pd.get_dummies(screenplays_model['category'])
rotten_dummies = pd.get_dummies(rotten_model['category'])
```


```python
# columns = ['sentiment_neutral', 'sentence_length', 'PRON', 'CCONJ', 'PUNCT', 'NNS',
#  '_SP', 'VBD', 'WDT', 'VB', 'PRP', 'RP', 'PRP$', 'CC', '.', 'IN', '-RRB-',
#  'VBP', 'WP', 'HYPH']

columns = ['word_count', 'unique_words']
columns.extend(rotten_model.columns[10:19])
columns.extend(rotten_model.columns[23:-2])

X = pd.merge(rotten_model[columns].copy(), rotten_dummies, left_index=True, 
             right_index=True)

y = rotten_model.good_or_bad.copy()
```


```python
X['predictions'] = model_1_predictions
```

Using the test data from previously, since it had not yet seen the overfit training model.


```python
fun.script_classifiers(X, y,
                   classifier=RandomForestClassifier(random_state=42,
                                                    max_depth=9,
                                                    criterion='entropy',
                                                    min_samples_split=14,
                                                    min_samples_leaf=1,
                                                    n_estimators=103),
                      use_tfidf=False, test_size=.2)
```

                  precision    recall  f1-score   support
    
               0       0.67      0.72      0.69       152
               1       0.70      0.64      0.67       154
    
        accuracy                           0.68       306
       macro avg       0.68      0.68      0.68       306
    weighted avg       0.68      0.68      0.68       306
    



![png](main_files/main_203_1.png)


This turned out to be my best model: SVC using TFIDF, then take the predictions from that, add them as a feature (to test data that has not been modeled) and run that data again through a random forest classifier: 68% accuracy, with a true positive rate of 64% and a true negative rate of 72%. Besides having the greatest accuracy, this model also turned out to be the most balanced.

## Linear Regression with Rottentomatoes

Next I attempted some regression models using data from rottontomatoes.

The models here are only the ones I kept. The others had results that were basically no better than chance.

### TFIDF with XGBoost Regressor

I grid searched the hell out of this one, only to find that none of it mattered. So I removed the grid search function.


```python
X_train, X_test, y_train, y_test = train_test_split(rotten_model.no_stop,
                                                    rotten_model.rotten_scores,
                                                    test_size=.3,
                                                    random_state=42)

tfidf = TfidfVectorizer(max_features=5000, max_df=.9, min_df=.1, 
                        ngram_range=(1,2))
word_predictors = tfidf.fit_transform(X_train)
word_test = tfidf.transform(X_test)

model = XGBRegressor(random_state=42, n_estimators=100, 
                     objective='reg:squarederror')
model.fit(word_predictors, y_train)
```

    //anaconda3/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning:
    
    Series.base is deprecated and will be removed in a future version
    





    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0,
                 importance_type='gain', learning_rate=0.1, max_delta_step=0,
                 max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:squarederror',
                 random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 seed=None, silent=None, subsample=1, verbosity=1)



r^2 and mse for train and test.


```python
def get_regression_scores(X_train, X_test, y_train_y_test, model):
    y_hat_train = model.predict(word_predictors)
    y_hat_test = model.predict(word_test)
    print('train MSE score: ', mse(y_train, y_hat_train))
    print('train r2_score: ', r2_score(y_train, y_hat_train))
    print('Test MSE score:', mse(y_test, y_hat_test))
    print('Test R-sq score:', r2_score(y_test, y_hat_test))
```


```python
y_hat_train = model.predict(word_predictors)
y_hat_test = model.predict(word_test)
print('train MSE score: ', mse(y_train, y_hat_train))
print('train r2_score: ', r2_score(y_train, y_hat_train))
print('Test MSE score:', mse(y_test, y_hat_test))
print('Test R-sq score:', r2_score(y_test, y_hat_test))
```

    train MSE score:  169.995391951505
    train r2_score:  0.7976318013144681
    Test MSE score: 786.8399855309698
    Test R-sq score: 0.03388834695151299



```python
train_residuals = y_hat_train - list(y_train)
test_residuals = y_hat_test - list(y_test)
print('Ave deviation from actual: ', round(sum(abs(test_residuals)) / len(test_residuals),2))
```

    Ave deviation from actual:  23.79


Creating a dataframe comparing predicted scores to actual scores.


```python
actual_v_predicted = X_test.to_frame()

pred_scores = list(y_hat_test)

actual_v_predicted['predicted_scores'] = pred_scores

to_merge = rotten_model[['titles', 'rotten_scores']]

actual_v_predicted = actual_v_predicted.merge(to_merge, left_index=True,
                                              right_index=True)

actual_v_predicted = actual_v_predicted[['titles','rotten_scores',
                                         'predicted_scores']]

actual_v_predicted['predicted_scores'] = actual_v_predicted.predicted_scores.\
    apply(lambda x: int(x))
```


```python
actual_v_predicted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>rotten_scores</th>
      <th>predicted_scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>655</th>
      <td>a-cure-for-wellness</td>
      <td>41</td>
      <td>46</td>
    </tr>
    <tr>
      <th>76</th>
      <td>big-mommas-like-father-like-son</td>
      <td>4</td>
      <td>38</td>
    </tr>
    <tr>
      <th>316</th>
      <td>all-eyez-on-me</td>
      <td>18</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1352</th>
      <td>pavarotti</td>
      <td>86</td>
      <td>61</td>
    </tr>
    <tr>
      <th>572</th>
      <td>state-like-sleep</td>
      <td>35</td>
      <td>49</td>
    </tr>
    <tr>
      <th>921</th>
      <td>despicable-me-3</td>
      <td>57</td>
      <td>43</td>
    </tr>
    <tr>
      <th>757</th>
      <td>goosebumps-2-haunted-halloween</td>
      <td>47</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1506</th>
      <td>mcqueen</td>
      <td>98</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>come-to-daddy</td>
      <td>86</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>serendipity</td>
      <td>76</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>pokemon-detective-pikachu</td>
      <td>68</td>
      <td>50</td>
    </tr>
    <tr>
      <th>877</th>
      <td>lego-ninjago-movie-the</td>
      <td>54</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1275</th>
      <td>mid90s</td>
      <td>81</td>
      <td>64</td>
    </tr>
    <tr>
      <th>1524</th>
      <td>fare-the</td>
      <td>99</td>
      <td>37</td>
    </tr>
    <tr>
      <th>988</th>
      <td>st-agatha</td>
      <td>62</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1419</th>
      <td>ghostbox-cowboy</td>
      <td>91</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1504</th>
      <td>i-am-not-your-negro</td>
      <td>98</td>
      <td>60</td>
    </tr>
    <tr>
      <th>30</th>
      <td>mortal-kombat-annihilation</td>
      <td>1</td>
      <td>46</td>
    </tr>
    <tr>
      <th>49</th>
      <td>battlefield-earth</td>
      <td>2</td>
      <td>34</td>
    </tr>
    <tr>
      <th>240</th>
      <td>man-down</td>
      <td>14</td>
      <td>46</td>
    </tr>
    <tr>
      <th>309</th>
      <td>euphoria</td>
      <td>18</td>
      <td>47</td>
    </tr>
    <tr>
      <th>352</th>
      <td>222</td>
      <td>21</td>
      <td>35</td>
    </tr>
    <tr>
      <th>124</th>
      <td>reprisal</td>
      <td>7</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>bernard-and-huey</td>
      <td>73</td>
      <td>54</td>
    </tr>
    <tr>
      <th>651</th>
      <td>phoenix-forgotten</td>
      <td>41</td>
      <td>46</td>
    </tr>
    <tr>
      <th>70</th>
      <td>vampires-suck</td>
      <td>3</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1503</th>
      <td>parasite</td>
      <td>98</td>
      <td>27</td>
    </tr>
    <tr>
      <th>703</th>
      <td>hurricane-heist-the</td>
      <td>44</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>paterno</td>
      <td>69</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1321</th>
      <td>await-further-instructions</td>
      <td>84</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1107</th>
      <td>mule-the</td>
      <td>69</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1318</th>
      <td>once-upon-a-time-in-hollywood</td>
      <td>84</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>border</td>
      <td>96</td>
      <td>55</td>
    </tr>
    <tr>
      <th>584</th>
      <td>blue-iguana</td>
      <td>36</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1220</th>
      <td>devils-doorway-the</td>
      <td>77</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>letters-from-baghdad</td>
      <td>83</td>
      <td>55</td>
    </tr>
    <tr>
      <th>500</th>
      <td>kin</td>
      <td>31</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1405</th>
      <td>a-star-is-born</td>
      <td>90</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1192</th>
      <td>dont-worry-he-wont-get-far-on-foot</td>
      <td>75</td>
      <td>52</td>
    </tr>
    <tr>
      <th>485</th>
      <td>pitch-perfect-3</td>
      <td>29</td>
      <td>42</td>
    </tr>
    <tr>
      <th>885</th>
      <td>first-purge-the</td>
      <td>54</td>
      <td>55</td>
    </tr>
    <tr>
      <th>952</th>
      <td>doom-annihilation</td>
      <td>59</td>
      <td>30</td>
    </tr>
    <tr>
      <th>611</th>
      <td>anon</td>
      <td>37</td>
      <td>48</td>
    </tr>
    <tr>
      <th>490</th>
      <td>secret-obsession</td>
      <td>30</td>
      <td>28</td>
    </tr>
    <tr>
      <th>353</th>
      <td>lucy-in-the-sky</td>
      <td>21</td>
      <td>58</td>
    </tr>
    <tr>
      <th>261</th>
      <td>fanatic-the</td>
      <td>15</td>
      <td>37</td>
    </tr>
    <tr>
      <th>59</th>
      <td>6-souls</td>
      <td>3</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>great-gilly-hopkins-the</td>
      <td>65</td>
      <td>32</td>
    </tr>
    <tr>
      <th>539</th>
      <td>last-weekend</td>
      <td>34</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>book-of-monsters</td>
      <td>74</td>
      <td>48</td>
    </tr>
    <tr>
      <th>259</th>
      <td>show-dogs</td>
      <td>15</td>
      <td>50</td>
    </tr>
    <tr>
      <th>768</th>
      <td>king-cobra</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1501</th>
      <td>ash-is-purest-white</td>
      <td>98</td>
      <td>66</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>angel-of-mine</td>
      <td>69</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1191</th>
      <td>hotel-mumbai</td>
      <td>75</td>
      <td>47</td>
    </tr>
    <tr>
      <th>462</th>
      <td>passage-to-mars</td>
      <td>28</td>
      <td>42</td>
    </tr>
    <tr>
      <th>51</th>
      <td>apparition-the</td>
      <td>2</td>
      <td>65</td>
    </tr>
    <tr>
      <th>471</th>
      <td>good-neighbor-the</td>
      <td>28</td>
      <td>25</td>
    </tr>
    <tr>
      <th>344</th>
      <td>boss-the</td>
      <td>21</td>
      <td>55</td>
    </tr>
    <tr>
      <th>679</th>
      <td>white-chamber</td>
      <td>42</td>
      <td>48</td>
    </tr>
    <tr>
      <th>303</th>
      <td>incarnate</td>
      <td>18</td>
      <td>23</td>
    </tr>
    <tr>
      <th>871</th>
      <td>jason-bourne</td>
      <td>53</td>
      <td>49</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dark-crimes</td>
      <td>0</td>
      <td>46</td>
    </tr>
    <tr>
      <th>590</th>
      <td>close</td>
      <td>36</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1413</th>
      <td>maria-by-callas</td>
      <td>91</td>
      <td>66</td>
    </tr>
    <tr>
      <th>461</th>
      <td>sunset</td>
      <td>28</td>
      <td>54</td>
    </tr>
    <tr>
      <th>861</th>
      <td>a-kid-like-jake</td>
      <td>52</td>
      <td>60</td>
    </tr>
    <tr>
      <th>767</th>
      <td>eli</td>
      <td>47</td>
      <td>47</td>
    </tr>
    <tr>
      <th>762</th>
      <td>kung-fu-yoga</td>
      <td>47</td>
      <td>49</td>
    </tr>
    <tr>
      <th>168</th>
      <td>all-relative</td>
      <td>9</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1483</th>
      <td>toy-story-4</td>
      <td>96</td>
      <td>40</td>
    </tr>
    <tr>
      <th>561</th>
      <td>6-underground</td>
      <td>35</td>
      <td>48</td>
    </tr>
    <tr>
      <th>568</th>
      <td>honeyglue</td>
      <td>35</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1482</th>
      <td>hate-u-give-the</td>
      <td>96</td>
      <td>42</td>
    </tr>
    <tr>
      <th>237</th>
      <td>vigilante-diaries</td>
      <td>14</td>
      <td>46</td>
    </tr>
    <tr>
      <th>910</th>
      <td>hummingbird-project-the</td>
      <td>56</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>tribe-the</td>
      <td>70</td>
      <td>54</td>
    </tr>
    <tr>
      <th>420</th>
      <td>unlocked</td>
      <td>25</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>12th-man-the</td>
      <td>85</td>
      <td>48</td>
    </tr>
    <tr>
      <th>99</th>
      <td>so-undercover</td>
      <td>5</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1527</th>
      <td>iron-fists-and-kung-fu-kicks</td>
      <td>99</td>
      <td>70</td>
    </tr>
    <tr>
      <th>827</th>
      <td>last-movie-star-the</td>
      <td>51</td>
      <td>40</td>
    </tr>
    <tr>
      <th>415</th>
      <td>killerman</td>
      <td>25</td>
      <td>42</td>
    </tr>
    <tr>
      <th>67</th>
      <td>flatliners</td>
      <td>3</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1109</th>
      <td>kill-team-the</td>
      <td>69</td>
      <td>38</td>
    </tr>
    <tr>
      <th>247</th>
      <td>lets-be-evil</td>
      <td>14</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>moss</td>
      <td>74</td>
      <td>51</td>
    </tr>
    <tr>
      <th>791</th>
      <td>superfly</td>
      <td>49</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>madelines-madeline</td>
      <td>87</td>
      <td>55</td>
    </tr>
    <tr>
      <th>841</th>
      <td>miracle-season-the</td>
      <td>51</td>
      <td>55</td>
    </tr>
    <tr>
      <th>555</th>
      <td>mark-felt-the-man-who-brought-down-the-white-h...</td>
      <td>34</td>
      <td>57</td>
    </tr>
    <tr>
      <th>812</th>
      <td>angelfish</td>
      <td>49</td>
      <td>48</td>
    </tr>
    <tr>
      <th>123</th>
      <td>grown-ups-2</td>
      <td>6</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>ready-player-one</td>
      <td>71</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1388</th>
      <td>souvenir-the</td>
      <td>89</td>
      <td>62</td>
    </tr>
    <tr>
      <th>422</th>
      <td>uglydolls</td>
      <td>26</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>lizzie</td>
      <td>66</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1125</th>
      <td>deadtectives</td>
      <td>70</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>stellas-last-weekend</td>
      <td>64</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>blood-on-her-name</td>
      <td>94</td>
      <td>40</td>
    </tr>
    <tr>
      <th>987</th>
      <td>sicario-day-of-the-soldado</td>
      <td>62</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1264</th>
      <td>daniel-isnt-real</td>
      <td>80</td>
      <td>56</td>
    </tr>
    <tr>
      <th>239</th>
      <td>a-score-to-settle</td>
      <td>14</td>
      <td>39</td>
    </tr>
    <tr>
      <th>367</th>
      <td>1517-to-paris-the</td>
      <td>22</td>
      <td>51</td>
    </tr>
    <tr>
      <th>481</th>
      <td>red-joan</td>
      <td>29</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1525</th>
      <td>after-we-leave</td>
      <td>99</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1404</th>
      <td>luce</td>
      <td>90</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1518</th>
      <td>tower</td>
      <td>98</td>
      <td>66</td>
    </tr>
    <tr>
      <th>993</th>
      <td>marie-curie-the-courage-of-knowledge</td>
      <td>62</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>use-me</td>
      <td>85</td>
      <td>67</td>
    </tr>
    <tr>
      <th>29</th>
      <td>meet-the-spartans</td>
      <td>1</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>suspiria</td>
      <td>64</td>
      <td>53</td>
    </tr>
    <tr>
      <th>342</th>
      <td>book-of-henry-the</td>
      <td>21</td>
      <td>42</td>
    </tr>
    <tr>
      <th>849</th>
      <td>roman-j-israel-esq</td>
      <td>52</td>
      <td>56</td>
    </tr>
    <tr>
      <th>416</th>
      <td>countdown</td>
      <td>25</td>
      <td>38</td>
    </tr>
    <tr>
      <th>65</th>
      <td>a-little-bit-of-heaven</td>
      <td>3</td>
      <td>55</td>
    </tr>
    <tr>
      <th>358</th>
      <td>adderall-diaries-the</td>
      <td>21</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1237</th>
      <td>rondo</td>
      <td>79</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>instant-family</td>
      <td>80</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1038</th>
      <td>little-stranger-the</td>
      <td>65</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>tales-from-the-hood-2</td>
      <td>74</td>
      <td>46</td>
    </tr>
    <tr>
      <th>141</th>
      <td>exposed</td>
      <td>7</td>
      <td>44</td>
    </tr>
    <tr>
      <th>802</th>
      <td>battle-of-jangsari-the</td>
      <td>49</td>
      <td>52</td>
    </tr>
    <tr>
      <th>953</th>
      <td>new-money</td>
      <td>59</td>
      <td>54</td>
    </tr>
    <tr>
      <th>514</th>
      <td>chamber-the</td>
      <td>32</td>
      <td>44</td>
    </tr>
    <tr>
      <th>727</th>
      <td>message-from-the-king</td>
      <td>45</td>
      <td>40</td>
    </tr>
    <tr>
      <th>275</th>
      <td>kissing-booth-the</td>
      <td>16</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1279</th>
      <td>boy-erased</td>
      <td>81</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>roxanne-roxanne</td>
      <td>69</td>
      <td>59</td>
    </tr>
    <tr>
      <th>513</th>
      <td>home-again</td>
      <td>32</td>
      <td>52</td>
    </tr>
    <tr>
      <th>627</th>
      <td>nostalgia</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1142</th>
      <td>a-christmas-prince</td>
      <td>72</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1046</th>
      <td>prophet-the</td>
      <td>65</td>
      <td>50</td>
    </tr>
    <tr>
      <th>928</th>
      <td>liam-gallagher-as-it-was</td>
      <td>58</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>rockaway</td>
      <td>63</td>
      <td>51</td>
    </tr>
    <tr>
      <th>44</th>
      <td>half-past-dead</td>
      <td>2</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>someone-great</td>
      <td>81</td>
      <td>59</td>
    </tr>
    <tr>
      <th>270</th>
      <td>looking-glass</td>
      <td>16</td>
      <td>45</td>
    </tr>
    <tr>
      <th>405</th>
      <td>beast-of-burden</td>
      <td>24</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>oceans-8</td>
      <td>68</td>
      <td>57</td>
    </tr>
    <tr>
      <th>115</th>
      <td>paranoia</td>
      <td>6</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>white-crow-the</td>
      <td>66</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>i-think-were-alone-now</td>
      <td>62</td>
      <td>42</td>
    </tr>
    <tr>
      <th>795</th>
      <td>cuck</td>
      <td>49</td>
      <td>42</td>
    </tr>
    <tr>
      <th>528</th>
      <td>an-actor-prepares</td>
      <td>32</td>
      <td>61</td>
    </tr>
    <tr>
      <th>432</th>
      <td>sherlock-gnomes</td>
      <td>26</td>
      <td>47</td>
    </tr>
    <tr>
      <th>896</th>
      <td>beach-bum-the</td>
      <td>55</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1200</th>
      <td>braven</td>
      <td>76</td>
      <td>44</td>
    </tr>
    <tr>
      <th>78</th>
      <td>hangman</td>
      <td>4</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>anthony-jeselnik-fire-in-the-maternity-ward</td>
      <td>79</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>mike-birbiglia-the-new-one</td>
      <td>99</td>
      <td>55</td>
    </tr>
    <tr>
      <th>939</th>
      <td>line-of-duty</td>
      <td>58</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1084</th>
      <td>journey-the</td>
      <td>68</td>
      <td>54</td>
    </tr>
    <tr>
      <th>630</th>
      <td>gringo</td>
      <td>39</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1045</th>
      <td>polka-king-the</td>
      <td>65</td>
      <td>47</td>
    </tr>
    <tr>
      <th>15</th>
      <td>dead-water</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1121</th>
      <td>black-site</td>
      <td>70</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1030</th>
      <td>resistance</td>
      <td>64</td>
      <td>62</td>
    </tr>
    <tr>
      <th>175</th>
      <td>last-heist-the</td>
      <td>10</td>
      <td>39</td>
    </tr>
    <tr>
      <th>989</th>
      <td>oath-the</td>
      <td>62</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>vanishing-the</td>
      <td>84</td>
      <td>42</td>
    </tr>
    <tr>
      <th>868</th>
      <td>crash-pad</td>
      <td>53</td>
      <td>29</td>
    </tr>
    <tr>
      <th>506</th>
      <td>intruder-the</td>
      <td>31</td>
      <td>36</td>
    </tr>
    <tr>
      <th>943</th>
      <td>ophelia</td>
      <td>59</td>
      <td>55</td>
    </tr>
    <tr>
      <th>365</th>
      <td>bad-santa-2</td>
      <td>22</td>
      <td>45</td>
    </tr>
    <tr>
      <th>549</th>
      <td>live-by-night</td>
      <td>34</td>
      <td>35</td>
    </tr>
    <tr>
      <th>203</th>
      <td>viking-destiny</td>
      <td>12</td>
      <td>50</td>
    </tr>
    <tr>
      <th>783</th>
      <td>churchill</td>
      <td>48</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>dark-river</td>
      <td>78</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>creed-ii</td>
      <td>83</td>
      <td>61</td>
    </tr>
    <tr>
      <th>1266</th>
      <td>judy</td>
      <td>81</td>
      <td>25</td>
    </tr>
    <tr>
      <th>811</th>
      <td>pimp</td>
      <td>49</td>
      <td>45</td>
    </tr>
    <tr>
      <th>913</th>
      <td>ray-romano-right-here-around-the-corner</td>
      <td>56</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1392</th>
      <td>lighthouse-the</td>
      <td>89</td>
      <td>53</td>
    </tr>
    <tr>
      <th>184</th>
      <td>ghost-team</td>
      <td>10</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>bikram-yogi-guru-predator</td>
      <td>95</td>
      <td>69</td>
    </tr>
    <tr>
      <th>479</th>
      <td>king-arthur-legend-of-the-sword</td>
      <td>29</td>
      <td>49</td>
    </tr>
    <tr>
      <th>56</th>
      <td>down-to-you</td>
      <td>2</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>batman-gotham-by-gaslight</td>
      <td>72</td>
      <td>56</td>
    </tr>
    <tr>
      <th>486</th>
      <td>rodin</td>
      <td>29</td>
      <td>48</td>
    </tr>
    <tr>
      <th>383</th>
      <td>kill-your-friends</td>
      <td>23</td>
      <td>57</td>
    </tr>
    <tr>
      <th>43</th>
      <td>passion-play</td>
      <td>2</td>
      <td>43</td>
    </tr>
    <tr>
      <th>533</th>
      <td>poms</td>
      <td>33</td>
      <td>59</td>
    </tr>
    <tr>
      <th>758</th>
      <td>going-in-style</td>
      <td>47</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>wonders-of-the-sea</td>
      <td>75</td>
      <td>44</td>
    </tr>
    <tr>
      <th>597</th>
      <td>i-am-vengeance</td>
      <td>37</td>
      <td>33</td>
    </tr>
    <tr>
      <th>101</th>
      <td>crucifixion-the</td>
      <td>5</td>
      <td>54</td>
    </tr>
    <tr>
      <th>552</th>
      <td>welcome-to-marwen</td>
      <td>34</td>
      <td>40</td>
    </tr>
    <tr>
      <th>579</th>
      <td>wetlands</td>
      <td>35</td>
      <td>47</td>
    </tr>
    <tr>
      <th>107</th>
      <td>fifty-shades-of-black</td>
      <td>6</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>coyote-lake</td>
      <td>82</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>zombieland-double-tap</td>
      <td>67</td>
      <td>43</td>
    </tr>
    <tr>
      <th>220</th>
      <td>when-the-bough-breaks</td>
      <td>12</td>
      <td>28</td>
    </tr>
    <tr>
      <th>946</th>
      <td>operation-finale</td>
      <td>59</td>
      <td>36</td>
    </tr>
    <tr>
      <th>451</th>
      <td>batman-v-superman-dawn-of-justice</td>
      <td>27</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1502</th>
      <td>moonlight</td>
      <td>98</td>
      <td>64</td>
    </tr>
    <tr>
      <th>889</th>
      <td>commuter-the</td>
      <td>55</td>
      <td>35</td>
    </tr>
    <tr>
      <th>725</th>
      <td>paul-apostle-of-christ</td>
      <td>45</td>
      <td>48</td>
    </tr>
    <tr>
      <th>128</th>
      <td>always-woodstock</td>
      <td>7</td>
      <td>52</td>
    </tr>
    <tr>
      <th>332</th>
      <td>terminal</td>
      <td>19</td>
      <td>38</td>
    </tr>
    <tr>
      <th>297</th>
      <td>chips</td>
      <td>18</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1466</th>
      <td>blackkklansman</td>
      <td>95</td>
      <td>26</td>
    </tr>
    <tr>
      <th>706</th>
      <td>jackals</td>
      <td>44</td>
      <td>45</td>
    </tr>
    <tr>
      <th>680</th>
      <td>15-minutes-of-war</td>
      <td>42</td>
      <td>51</td>
    </tr>
    <tr>
      <th>244</th>
      <td>transformers-the-last-knight</td>
      <td>14</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1522</th>
      <td>parting-glass-the</td>
      <td>99</td>
      <td>41</td>
    </tr>
    <tr>
      <th>662</th>
      <td>valley-the</td>
      <td>41</td>
      <td>50</td>
    </tr>
    <tr>
      <th>847</th>
      <td>lion-king-the</td>
      <td>52</td>
      <td>36</td>
    </tr>
    <tr>
      <th>708</th>
      <td>girls-of-the-sun</td>
      <td>44</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1171</th>
      <td>after-louie</td>
      <td>74</td>
      <td>44</td>
    </tr>
    <tr>
      <th>231</th>
      <td>circle-the</td>
      <td>13</td>
      <td>42</td>
    </tr>
    <tr>
      <th>32</th>
      <td>daddy-day-camp</td>
      <td>1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1201</th>
      <td>i-kill-giants</td>
      <td>76</td>
      <td>50</td>
    </tr>
    <tr>
      <th>494</th>
      <td>mapplethorpe</td>
      <td>30</td>
      <td>52</td>
    </tr>
    <tr>
      <th>599</th>
      <td>kidnap</td>
      <td>37</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>who-would-you-take-to-a-deserted-island</td>
      <td>82</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1185</th>
      <td>mirage</td>
      <td>74</td>
      <td>59</td>
    </tr>
    <tr>
      <th>350</th>
      <td>6-below-miracle-on-the-mountain</td>
      <td>21</td>
      <td>43</td>
    </tr>
    <tr>
      <th>992</th>
      <td>childs-play</td>
      <td>62</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1155</th>
      <td>on-the-basis-of-sex</td>
      <td>72</td>
      <td>41</td>
    </tr>
    <tr>
      <th>967</th>
      <td>hotel-transylvania-3-summer-vacation</td>
      <td>61</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>dumplin</td>
      <td>84</td>
      <td>43</td>
    </tr>
    <tr>
      <th>411</th>
      <td>unforgettable</td>
      <td>25</td>
      <td>48</td>
    </tr>
    <tr>
      <th>163</th>
      <td>stonewall</td>
      <td>9</td>
      <td>49</td>
    </tr>
    <tr>
      <th>906</th>
      <td>154</td>
      <td>56</td>
      <td>56</td>
    </tr>
    <tr>
      <th>497</th>
      <td>wonder-wheel</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>306</th>
      <td>possession-of-hannah-grace-the</td>
      <td>18</td>
      <td>50</td>
    </tr>
    <tr>
      <th>433</th>
      <td>escape-plan-the-extractors</td>
      <td>26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>671</th>
      <td>katie-says-goodbye</td>
      <td>41</td>
      <td>35</td>
    </tr>
    <tr>
      <th>600</th>
      <td>el-chicano</td>
      <td>37</td>
      <td>40</td>
    </tr>
    <tr>
      <th>371</th>
      <td>rim-of-the-world</td>
      <td>22</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1426</th>
      <td>fighting-with-my-family</td>
      <td>92</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>speed-of-life</td>
      <td>77</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1088</th>
      <td>final-score</td>
      <td>68</td>
      <td>56</td>
    </tr>
    <tr>
      <th>192</th>
      <td>basement-the</td>
      <td>10</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>triple-frontier</td>
      <td>70</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>terminator-dark-fate</td>
      <td>69</td>
      <td>40</td>
    </tr>
    <tr>
      <th>929</th>
      <td>astronaut</td>
      <td>58</td>
      <td>43</td>
    </tr>
    <tr>
      <th>63</th>
      <td>playing-for-keeps</td>
      <td>3</td>
      <td>46</td>
    </tr>
    <tr>
      <th>709</th>
      <td>demon-house</td>
      <td>44</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1150</th>
      <td>galveston</td>
      <td>72</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>thank-you-for-your-service</td>
      <td>76</td>
      <td>32</td>
    </tr>
    <tr>
      <th>942</th>
      <td>bad-moms</td>
      <td>58</td>
      <td>54</td>
    </tr>
    <tr>
      <th>911</th>
      <td>lbj</td>
      <td>56</td>
      <td>50</td>
    </tr>
    <tr>
      <th>274</th>
      <td>look-away</td>
      <td>16</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>prospect</td>
      <td>88</td>
      <td>49</td>
    </tr>
    <tr>
      <th>670</th>
      <td>black-butterfly</td>
      <td>41</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1382</th>
      <td>blinded-by-the-light</td>
      <td>88</td>
      <td>55</td>
    </tr>
    <tr>
      <th>198</th>
      <td>fifty-shades-freed</td>
      <td>11</td>
      <td>29</td>
    </tr>
    <tr>
      <th>453</th>
      <td>into-the-ashes</td>
      <td>27</td>
      <td>52</td>
    </tr>
    <tr>
      <th>339</th>
      <td>phil</td>
      <td>20</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1039</th>
      <td>danger-close</td>
      <td>65</td>
      <td>48</td>
    </tr>
    <tr>
      <th>892</th>
      <td>greatest-showman-the</td>
      <td>55</td>
      <td>44</td>
    </tr>
    <tr>
      <th>271</th>
      <td>last-summer-the</td>
      <td>16</td>
      <td>52</td>
    </tr>
    <tr>
      <th>577</th>
      <td>most-hated-woman-in-america-the</td>
      <td>35</td>
      <td>38</td>
    </tr>
    <tr>
      <th>218</th>
      <td>day-of-the-dead-bloodline</td>
      <td>12</td>
      <td>36</td>
    </tr>
    <tr>
      <th>799</th>
      <td>postcards-from-london</td>
      <td>49</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1241</th>
      <td>greener-grass</td>
      <td>79</td>
      <td>50</td>
    </tr>
    <tr>
      <th>427</th>
      <td>week-of-the</td>
      <td>26</td>
      <td>51</td>
    </tr>
    <tr>
      <th>351</th>
      <td>i-hate-kids</td>
      <td>21</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>before-i-fall</td>
      <td>64</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1267</th>
      <td>room-for-rent</td>
      <td>81</td>
      <td>54</td>
    </tr>
    <tr>
      <th>324</th>
      <td>beneath-the-leaves</td>
      <td>19</td>
      <td>49</td>
    </tr>
    <tr>
      <th>413</th>
      <td>undrafted</td>
      <td>25</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>what-they-had</td>
      <td>86</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1043</th>
      <td>body-at-brighton-rock</td>
      <td>65</td>
      <td>60</td>
    </tr>
    <tr>
      <th>696</th>
      <td>ready-to-mingle</td>
      <td>43</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>wrinkles-the-clown</td>
      <td>69</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>hale-county-this-morning-this-evening</td>
      <td>96</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>ghost-light</td>
      <td>73</td>
      <td>57</td>
    </tr>
    <tr>
      <th>289</th>
      <td>great-war-the</td>
      <td>16</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1173</th>
      <td>anchor-and-hope</td>
      <td>74</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1427</th>
      <td>last-race-the</td>
      <td>92</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>tribes-of-palos-verdes-the</td>
      <td>63</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>unicorn-store</td>
      <td>63</td>
      <td>62</td>
    </tr>
    <tr>
      <th>619</th>
      <td>slaughterhouse-rulez</td>
      <td>38</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>changeland</td>
      <td>62</td>
      <td>34</td>
    </tr>
    <tr>
      <th>277</th>
      <td>jeepers-creepers-3</td>
      <td>16</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>mega-time-squad</td>
      <td>75</td>
      <td>53</td>
    </tr>
    <tr>
      <th>864</th>
      <td>47-meters-down</td>
      <td>53</td>
      <td>56</td>
    </tr>
    <tr>
      <th>236</th>
      <td>open-house-the</td>
      <td>14</td>
      <td>58</td>
    </tr>
    <tr>
      <th>615</th>
      <td>whisky-galore</td>
      <td>38</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>bill-burr-paper-tiger</td>
      <td>85</td>
      <td>66</td>
    </tr>
    <tr>
      <th>1480</th>
      <td>gangster-the-cop-the-devil-the</td>
      <td>96</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1402</th>
      <td>mirai</td>
      <td>90</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>downton-abbey</td>
      <td>83</td>
      <td>51</td>
    </tr>
    <tr>
      <th>208</th>
      <td>sword-of-vengeance</td>
      <td>12</td>
      <td>58</td>
    </tr>
    <tr>
      <th>380</th>
      <td>mr-church</td>
      <td>23</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1252</th>
      <td>kevin-harts-guide-to-black-history</td>
      <td>79</td>
      <td>52</td>
    </tr>
    <tr>
      <th>941</th>
      <td>absolutely-fabulous-the-movie</td>
      <td>58</td>
      <td>39</td>
    </tr>
    <tr>
      <th>58</th>
      <td>happily-never-after</td>
      <td>3</td>
      <td>64</td>
    </tr>
    <tr>
      <th>382</th>
      <td>90-minutes-in-heaven</td>
      <td>23</td>
      <td>48</td>
    </tr>
    <tr>
      <th>374</th>
      <td>clapper-the</td>
      <td>22</td>
      <td>41</td>
    </tr>
    <tr>
      <th>429</th>
      <td>rambo-last-blood</td>
      <td>26</td>
      <td>55</td>
    </tr>
    <tr>
      <th>997</th>
      <td>it-chapter-two</td>
      <td>62</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1230</th>
      <td>buffalo-boys</td>
      <td>78</td>
      <td>34</td>
    </tr>
    <tr>
      <th>744</th>
      <td>hollars-the</td>
      <td>46</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1240</th>
      <td>30-miles-from-nowhere</td>
      <td>79</td>
      <td>32</td>
    </tr>
    <tr>
      <th>361</th>
      <td>ithaca</td>
      <td>21</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1083</th>
      <td>disappearance-at-clifton-hill</td>
      <td>68</td>
      <td>53</td>
    </tr>
    <tr>
      <th>464</th>
      <td>new-life</td>
      <td>28</td>
      <td>56</td>
    </tr>
    <tr>
      <th>583</th>
      <td>resident-evil-the-final-chapter</td>
      <td>36</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>spider-in-the-web</td>
      <td>74</td>
      <td>42</td>
    </tr>
    <tr>
      <th>844</th>
      <td>magic-in-the-moonlight</td>
      <td>51</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>adopt-a-highway</td>
      <td>67</td>
      <td>52</td>
    </tr>
    <tr>
      <th>233</th>
      <td>nine-lives</td>
      <td>14</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>mustang-the</td>
      <td>93</td>
      <td>61</td>
    </tr>
    <tr>
      <th>196</th>
      <td>point-break</td>
      <td>11</td>
      <td>50</td>
    </tr>
    <tr>
      <th>665</th>
      <td>aftermath</td>
      <td>41</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>afterward</td>
      <td>82</td>
      <td>21</td>
    </tr>
    <tr>
      <th>81</th>
      <td>abduction</td>
      <td>4</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1438</th>
      <td>screwball</td>
      <td>93</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>dog-days</td>
      <td>62</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>favourite-the</td>
      <td>92</td>
      <td>56</td>
    </tr>
    <tr>
      <th>111</th>
      <td>cbgb</td>
      <td>6</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>rust-creek</td>
      <td>83</td>
      <td>35</td>
    </tr>
    <tr>
      <th>809</th>
      <td>a-christmas-prince-the-royal-wedding</td>
      <td>49</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>day-shall-come-the</td>
      <td>64</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>avengers-endgame</td>
      <td>93</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>official-secrets</td>
      <td>82</td>
      <td>41</td>
    </tr>
    <tr>
      <th>676</th>
      <td>ghost-stories</td>
      <td>42</td>
      <td>59</td>
    </tr>
    <tr>
      <th>310</th>
      <td>diary-of-a-wimpy-kid-the-long-haul</td>
      <td>18</td>
      <td>47</td>
    </tr>
    <tr>
      <th>612</th>
      <td>social-animals</td>
      <td>37</td>
      <td>53</td>
    </tr>
    <tr>
      <th>925</th>
      <td>close-enemies</td>
      <td>57</td>
      <td>35</td>
    </tr>
    <tr>
      <th>920</th>
      <td>hippopotamus-the</td>
      <td>57</td>
      <td>53</td>
    </tr>
    <tr>
      <th>848</th>
      <td>american-dreamer</td>
      <td>52</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>snapshots</td>
      <td>71</td>
      <td>51</td>
    </tr>
    <tr>
      <th>620</th>
      <td>death-note</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>366</th>
      <td>mile-22</td>
      <td>22</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1523</th>
      <td>toxic-beauty</td>
      <td>99</td>
      <td>68</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>fire-in-paradise</td>
      <td>79</td>
      <td>55</td>
    </tr>
    <tr>
      <th>425</th>
      <td>victor-frankenstein</td>
      <td>26</td>
      <td>41</td>
    </tr>
    <tr>
      <th>884</th>
      <td>mrs-hyde</td>
      <td>54</td>
      <td>52</td>
    </tr>
    <tr>
      <th>832</th>
      <td>boss-baby-the</td>
      <td>51</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>a-private-war</td>
      <td>87</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1408</th>
      <td>how-to-train-your-dragon-the-hidden-world</td>
      <td>90</td>
      <td>42</td>
    </tr>
    <tr>
      <th>650</th>
      <td>spinning-man</td>
      <td>41</td>
      <td>45</td>
    </tr>
    <tr>
      <th>589</th>
      <td>leisure-seeker-the</td>
      <td>36</td>
      <td>57</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>last-flag-flying</td>
      <td>77</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>sword-of-trust</td>
      <td>92</td>
      <td>43</td>
    </tr>
    <tr>
      <th>785</th>
      <td>freeheld</td>
      <td>48</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>lego-movie-2-the-second-part-the</td>
      <td>84</td>
      <td>53</td>
    </tr>
    <tr>
      <th>254</th>
      <td>sinister-2</td>
      <td>15</td>
      <td>26</td>
    </tr>
    <tr>
      <th>322</th>
      <td>hospitality</td>
      <td>19</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1118</th>
      <td>horse-girl</td>
      <td>70</td>
      <td>54</td>
    </tr>
    <tr>
      <th>428</th>
      <td>little-women</td>
      <td>26</td>
      <td>79</td>
    </tr>
    <tr>
      <th>576</th>
      <td>i-feel-pretty</td>
      <td>35</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1078</th>
      <td>puppet-master-the-littlest-reich</td>
      <td>67</td>
      <td>48</td>
    </tr>
    <tr>
      <th>872</th>
      <td>gemma-bovery</td>
      <td>53</td>
      <td>44</td>
    </tr>
    <tr>
      <th>354</th>
      <td>zoolander-2</td>
      <td>21</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>invader-zim-enter-the-florpus</td>
      <td>99</td>
      <td>52</td>
    </tr>
    <tr>
      <th>86</th>
      <td>i-frankenstein</td>
      <td>4</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>burning</td>
      <td>94</td>
      <td>52</td>
    </tr>
    <tr>
      <th>265</th>
      <td>mummy-the</td>
      <td>15</td>
      <td>28</td>
    </tr>
    <tr>
      <th>287</th>
      <td>dear-dictator</td>
      <td>16</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1519</th>
      <td>things-to-come</td>
      <td>98</td>
      <td>62</td>
    </tr>
    <tr>
      <th>438</th>
      <td>drone</td>
      <td>26</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>premature</td>
      <td>92</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1089</th>
      <td>housemaid-the</td>
      <td>68</td>
      <td>45</td>
    </tr>
    <tr>
      <th>439</th>
      <td>duel-the</td>
      <td>26</td>
      <td>45</td>
    </tr>
    <tr>
      <th>759</th>
      <td>war-machine</td>
      <td>47</td>
      <td>45</td>
    </tr>
    <tr>
      <th>221</th>
      <td>american-heist</td>
      <td>12</td>
      <td>53</td>
    </tr>
    <tr>
      <th>113</th>
      <td>java-heat</td>
      <td>6</td>
      <td>33</td>
    </tr>
    <tr>
      <th>591</th>
      <td>in-the-tall-grass</td>
      <td>36</td>
      <td>40</td>
    </tr>
    <tr>
      <th>602</th>
      <td>backstabbing-for-beginners</td>
      <td>37</td>
      <td>59</td>
    </tr>
    <tr>
      <th>174</th>
      <td>vanishing-of-sidney-hall-the</td>
      <td>10</td>
      <td>40</td>
    </tr>
    <tr>
      <th>553</th>
      <td>vhs-viral</td>
      <td>34</td>
      <td>51</td>
    </tr>
    <tr>
      <th>381</th>
      <td>nobodys-fool</td>
      <td>23</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>standoff-at-sparrow-creek-the</td>
      <td>76</td>
      <td>61</td>
    </tr>
    <tr>
      <th>18</th>
      <td>london-fields</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1194</th>
      <td>in-the-fade</td>
      <td>76</td>
      <td>50</td>
    </tr>
    <tr>
      <th>285</th>
      <td>competition-the</td>
      <td>16</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>black-47</td>
      <td>77</td>
      <td>66</td>
    </tr>
    <tr>
      <th>749</th>
      <td>clown</td>
      <td>46</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>revival-the</td>
      <td>78</td>
      <td>47</td>
    </tr>
    <tr>
      <th>31</th>
      <td>baby-geniuses</td>
      <td>1</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1165</th>
      <td>claras-ghost</td>
      <td>73</td>
      <td>54</td>
    </tr>
    <tr>
      <th>210</th>
      <td>mortdecai</td>
      <td>12</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>lasso</td>
      <td>62</td>
      <td>43</td>
    </tr>
    <tr>
      <th>478</th>
      <td>outlaws</td>
      <td>29</td>
      <td>56</td>
    </tr>
    <tr>
      <th>839</th>
      <td>rampage</td>
      <td>51</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2-in-the-bush-a-love-story</td>
      <td>99</td>
      <td>50</td>
    </tr>
    <tr>
      <th>965</th>
      <td>alita-battle-angel</td>
      <td>60</td>
      <td>32</td>
    </tr>
    <tr>
      <th>394</th>
      <td>rememory</td>
      <td>24</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>una</td>
      <td>76</td>
      <td>51</td>
    </tr>
    <tr>
      <th>188</th>
      <td>great-alaskan-race-the</td>
      <td>10</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>angel-the</td>
      <td>74</td>
      <td>48</td>
    </tr>
    <tr>
      <th>969</th>
      <td>mary-queen-of-scots</td>
      <td>61</td>
      <td>51</td>
    </tr>
    <tr>
      <th>635</th>
      <td>last-witness-the</td>
      <td>39</td>
      <td>46</td>
    </tr>
    <tr>
      <th>493</th>
      <td>loving-pablo</td>
      <td>30</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>hustlers</td>
      <td>86</td>
      <td>57</td>
    </tr>
    <tr>
      <th>934</th>
      <td>white-boy-rick</td>
      <td>58</td>
      <td>42</td>
    </tr>
    <tr>
      <th>83</th>
      <td>home-sweet-hell</td>
      <td>4</td>
      <td>28</td>
    </tr>
    <tr>
      <th>695</th>
      <td>wedding-guest-the</td>
      <td>43</td>
      <td>34</td>
    </tr>
    <tr>
      <th>48</th>
      <td>bless-the-child</td>
      <td>2</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>long-dumb-road-the</td>
      <td>72</td>
      <td>49</td>
    </tr>
    <tr>
      <th>155</th>
      <td>dont-sleep</td>
      <td>8</td>
      <td>47</td>
    </tr>
    <tr>
      <th>426</th>
      <td>london-has-fallen</td>
      <td>26</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1167</th>
      <td>sobibor</td>
      <td>74</td>
      <td>52</td>
    </tr>
    <tr>
      <th>435</th>
      <td>mortal-engines</td>
      <td>26</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1076</th>
      <td>joker</td>
      <td>67</td>
      <td>50</td>
    </tr>
    <tr>
      <th>837</th>
      <td>tomb-raider</td>
      <td>51</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>vfw</td>
      <td>80</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1516</th>
      <td>gods-own-country</td>
      <td>98</td>
      <td>53</td>
    </tr>
    <tr>
      <th>536</th>
      <td>wonder-park</td>
      <td>33</td>
      <td>40</td>
    </tr>
    <tr>
      <th>179</th>
      <td>i-am-wrath</td>
      <td>10</td>
      <td>47</td>
    </tr>
    <tr>
      <th>544</th>
      <td>abattoir</td>
      <td>34</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1278</th>
      <td>level-16</td>
      <td>81</td>
      <td>62</td>
    </tr>
    <tr>
      <th>632</th>
      <td>run-the-race</td>
      <td>39</td>
      <td>44</td>
    </tr>
    <tr>
      <th>717</th>
      <td>everything-everything</td>
      <td>45</td>
      <td>53</td>
    </tr>
    <tr>
      <th>846</th>
      <td>genius</td>
      <td>52</td>
      <td>48</td>
    </tr>
    <tr>
      <th>363</th>
      <td>overdrive</td>
      <td>22</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1205</th>
      <td>jumanji-welcome-to-the-jungle</td>
      <td>76</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1228</th>
      <td>all-the-money-in-the-world</td>
      <td>78</td>
      <td>43</td>
    </tr>
    <tr>
      <th>346</th>
      <td>14-cameras</td>
      <td>21</td>
      <td>53</td>
    </tr>
    <tr>
      <th>821</th>
      <td>miss-julie</td>
      <td>51</td>
      <td>52</td>
    </tr>
    <tr>
      <th>535</th>
      <td>fun-mom-dinner</td>
      <td>33</td>
      <td>62</td>
    </tr>
    <tr>
      <th>681</th>
      <td>maze-runner-the-death-cure</td>
      <td>42</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1325</th>
      <td>untouchable</td>
      <td>84</td>
      <td>55</td>
    </tr>
    <tr>
      <th>608</th>
      <td>along-came-the-devil</td>
      <td>37</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1274</th>
      <td>ben-is-back</td>
      <td>81</td>
      <td>55</td>
    </tr>
    <tr>
      <th>944</th>
      <td>scarborough</td>
      <td>59</td>
      <td>52</td>
    </tr>
    <tr>
      <th>243</th>
      <td>sharknado-the-4th-awakens</td>
      <td>14</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1491</th>
      <td>homecoming-a-film-by-beyonce</td>
      <td>97</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1430</th>
      <td>high-flying-bird</td>
      <td>92</td>
      <td>58</td>
    </tr>
    <tr>
      <th>637</th>
      <td>girl-in-the-spiders-web-the</td>
      <td>39</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>wildling</td>
      <td>71</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>all-the-bright-places</td>
      <td>75</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>68-kill</td>
      <td>68</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>private-life</td>
      <td>94</td>
      <td>46</td>
    </tr>
    <tr>
      <th>815</th>
      <td>an-evening-with-beverly-luff-linn</td>
      <td>50</td>
      <td>44</td>
    </tr>
    <tr>
      <th>308</th>
      <td>underworld-blood-wars</td>
      <td>18</td>
      <td>39</td>
    </tr>
    <tr>
      <th>430</th>
      <td>can-you-keep-a-secret</td>
      <td>26</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>solo-a-star-wars-story</td>
      <td>69</td>
      <td>37</td>
    </tr>
    <tr>
      <th>398</th>
      <td>otherhood</td>
      <td>24</td>
      <td>47</td>
    </tr>
    <tr>
      <th>423</th>
      <td>hellions</td>
      <td>26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>370</th>
      <td>dark-phoenix</td>
      <td>22</td>
      <td>49</td>
    </tr>
    <tr>
      <th>54</th>
      <td>deuces-wild</td>
      <td>2</td>
      <td>40</td>
    </tr>
    <tr>
      <th>790</th>
      <td>sonata-the</td>
      <td>49</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1169</th>
      <td>monster-party</td>
      <td>74</td>
      <td>40</td>
    </tr>
    <tr>
      <th>996</th>
      <td>bird-box</td>
      <td>62</td>
      <td>43</td>
    </tr>
    <tr>
      <th>312</th>
      <td>blind</td>
      <td>18</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>first-man</td>
      <td>86</td>
      <td>58</td>
    </tr>
    <tr>
      <th>592</th>
      <td>morgan</td>
      <td>36</td>
      <td>44</td>
    </tr>
    <tr>
      <th>109</th>
      <td>amityville-murders-the</td>
      <td>6</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>changeover-the</td>
      <td>68</td>
      <td>43</td>
    </tr>
    <tr>
      <th>813</th>
      <td>escape-room</td>
      <td>49</td>
      <td>36</td>
    </tr>
    <tr>
      <th>100</th>
      <td>extraction</td>
      <td>5</td>
      <td>38</td>
    </tr>
    <tr>
      <th>482</th>
      <td>leatherface</td>
      <td>29</td>
      <td>32</td>
    </tr>
    <tr>
      <th>693</th>
      <td>tall-girl</td>
      <td>43</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1409</th>
      <td>rolling-thunder-revue-a-bob-dylan-story-by-mar...</td>
      <td>91</td>
      <td>76</td>
    </tr>
    <tr>
      <th>266</th>
      <td>transporter-refueled-the</td>
      <td>15</td>
      <td>28</td>
    </tr>
    <tr>
      <th>702</th>
      <td>red-sparrow</td>
      <td>44</td>
      <td>35</td>
    </tr>
    <tr>
      <th>296</th>
      <td>god-bless-the-broken-road</td>
      <td>17</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>colette</td>
      <td>86</td>
      <td>53</td>
    </tr>
    <tr>
      <th>977</th>
      <td>uncle-drew</td>
      <td>61</td>
      <td>43</td>
    </tr>
    <tr>
      <th>723</th>
      <td>meg-the</td>
      <td>45</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1208</th>
      <td>little-pink-house</td>
      <td>76</td>
      <td>57</td>
    </tr>
    <tr>
      <th>377</th>
      <td>birth-of-the-dragon</td>
      <td>23</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>




```python
upside = len(actual_v_predicted[(actual_v_predicted.rotten_scores > 50) \
                   & (actual_v_predicted.predicted_scores > 50)])
upside

downside = len(actual_v_predicted[(actual_v_predicted.rotten_scores < 50) \
                   & (actual_v_predicted.predicted_scores < 50)])
downside

all_sides = len(actual_v_predicted)
all_sides

capture_in_half = (upside + downside) / len(actual_v_predicted)
print('Scores captured in same half (upper vs lower): ', capture_in_half)
```

    Scores captured in same half (upper vs lower):  0.579520697167756



```python
actual_v_predicted.rotten_scores.hist();
```


![png](main_files/main_218_0.png)



```python
actual_v_predicted.predicted_scores.hist();
```


![png](main_files/main_219_0.png)



```python
fig = sm.graphics.qqplot(test_residuals, dist=stats.norm, line='45', fit=True)
fig.show();
```

    //anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: UserWarning:
    
    Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
    



![png](main_files/main_220_1.png)



```python
actual_v_predicted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>rotten_scores</th>
      <th>predicted_scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>655</th>
      <td>a-cure-for-wellness</td>
      <td>41</td>
      <td>46</td>
    </tr>
    <tr>
      <th>76</th>
      <td>big-mommas-like-father-like-son</td>
      <td>4</td>
      <td>38</td>
    </tr>
    <tr>
      <th>316</th>
      <td>all-eyez-on-me</td>
      <td>18</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1352</th>
      <td>pavarotti</td>
      <td>86</td>
      <td>61</td>
    </tr>
    <tr>
      <th>572</th>
      <td>state-like-sleep</td>
      <td>35</td>
      <td>49</td>
    </tr>
    <tr>
      <th>921</th>
      <td>despicable-me-3</td>
      <td>57</td>
      <td>43</td>
    </tr>
    <tr>
      <th>757</th>
      <td>goosebumps-2-haunted-halloween</td>
      <td>47</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1506</th>
      <td>mcqueen</td>
      <td>98</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>come-to-daddy</td>
      <td>86</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>serendipity</td>
      <td>76</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>pokemon-detective-pikachu</td>
      <td>68</td>
      <td>50</td>
    </tr>
    <tr>
      <th>877</th>
      <td>lego-ninjago-movie-the</td>
      <td>54</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1275</th>
      <td>mid90s</td>
      <td>81</td>
      <td>64</td>
    </tr>
    <tr>
      <th>1524</th>
      <td>fare-the</td>
      <td>99</td>
      <td>37</td>
    </tr>
    <tr>
      <th>988</th>
      <td>st-agatha</td>
      <td>62</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1419</th>
      <td>ghostbox-cowboy</td>
      <td>91</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1504</th>
      <td>i-am-not-your-negro</td>
      <td>98</td>
      <td>60</td>
    </tr>
    <tr>
      <th>30</th>
      <td>mortal-kombat-annihilation</td>
      <td>1</td>
      <td>46</td>
    </tr>
    <tr>
      <th>49</th>
      <td>battlefield-earth</td>
      <td>2</td>
      <td>34</td>
    </tr>
    <tr>
      <th>240</th>
      <td>man-down</td>
      <td>14</td>
      <td>46</td>
    </tr>
    <tr>
      <th>309</th>
      <td>euphoria</td>
      <td>18</td>
      <td>47</td>
    </tr>
    <tr>
      <th>352</th>
      <td>222</td>
      <td>21</td>
      <td>35</td>
    </tr>
    <tr>
      <th>124</th>
      <td>reprisal</td>
      <td>7</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>bernard-and-huey</td>
      <td>73</td>
      <td>54</td>
    </tr>
    <tr>
      <th>651</th>
      <td>phoenix-forgotten</td>
      <td>41</td>
      <td>46</td>
    </tr>
    <tr>
      <th>70</th>
      <td>vampires-suck</td>
      <td>3</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1503</th>
      <td>parasite</td>
      <td>98</td>
      <td>27</td>
    </tr>
    <tr>
      <th>703</th>
      <td>hurricane-heist-the</td>
      <td>44</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>paterno</td>
      <td>69</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1321</th>
      <td>await-further-instructions</td>
      <td>84</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1107</th>
      <td>mule-the</td>
      <td>69</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1318</th>
      <td>once-upon-a-time-in-hollywood</td>
      <td>84</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>border</td>
      <td>96</td>
      <td>55</td>
    </tr>
    <tr>
      <th>584</th>
      <td>blue-iguana</td>
      <td>36</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1220</th>
      <td>devils-doorway-the</td>
      <td>77</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>letters-from-baghdad</td>
      <td>83</td>
      <td>55</td>
    </tr>
    <tr>
      <th>500</th>
      <td>kin</td>
      <td>31</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1405</th>
      <td>a-star-is-born</td>
      <td>90</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1192</th>
      <td>dont-worry-he-wont-get-far-on-foot</td>
      <td>75</td>
      <td>52</td>
    </tr>
    <tr>
      <th>485</th>
      <td>pitch-perfect-3</td>
      <td>29</td>
      <td>42</td>
    </tr>
    <tr>
      <th>885</th>
      <td>first-purge-the</td>
      <td>54</td>
      <td>55</td>
    </tr>
    <tr>
      <th>952</th>
      <td>doom-annihilation</td>
      <td>59</td>
      <td>30</td>
    </tr>
    <tr>
      <th>611</th>
      <td>anon</td>
      <td>37</td>
      <td>48</td>
    </tr>
    <tr>
      <th>490</th>
      <td>secret-obsession</td>
      <td>30</td>
      <td>28</td>
    </tr>
    <tr>
      <th>353</th>
      <td>lucy-in-the-sky</td>
      <td>21</td>
      <td>58</td>
    </tr>
    <tr>
      <th>261</th>
      <td>fanatic-the</td>
      <td>15</td>
      <td>37</td>
    </tr>
    <tr>
      <th>59</th>
      <td>6-souls</td>
      <td>3</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>great-gilly-hopkins-the</td>
      <td>65</td>
      <td>32</td>
    </tr>
    <tr>
      <th>539</th>
      <td>last-weekend</td>
      <td>34</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>book-of-monsters</td>
      <td>74</td>
      <td>48</td>
    </tr>
    <tr>
      <th>259</th>
      <td>show-dogs</td>
      <td>15</td>
      <td>50</td>
    </tr>
    <tr>
      <th>768</th>
      <td>king-cobra</td>
      <td>48</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1501</th>
      <td>ash-is-purest-white</td>
      <td>98</td>
      <td>66</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>angel-of-mine</td>
      <td>69</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1191</th>
      <td>hotel-mumbai</td>
      <td>75</td>
      <td>47</td>
    </tr>
    <tr>
      <th>462</th>
      <td>passage-to-mars</td>
      <td>28</td>
      <td>42</td>
    </tr>
    <tr>
      <th>51</th>
      <td>apparition-the</td>
      <td>2</td>
      <td>65</td>
    </tr>
    <tr>
      <th>471</th>
      <td>good-neighbor-the</td>
      <td>28</td>
      <td>25</td>
    </tr>
    <tr>
      <th>344</th>
      <td>boss-the</td>
      <td>21</td>
      <td>55</td>
    </tr>
    <tr>
      <th>679</th>
      <td>white-chamber</td>
      <td>42</td>
      <td>48</td>
    </tr>
    <tr>
      <th>303</th>
      <td>incarnate</td>
      <td>18</td>
      <td>23</td>
    </tr>
    <tr>
      <th>871</th>
      <td>jason-bourne</td>
      <td>53</td>
      <td>49</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dark-crimes</td>
      <td>0</td>
      <td>46</td>
    </tr>
    <tr>
      <th>590</th>
      <td>close</td>
      <td>36</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1413</th>
      <td>maria-by-callas</td>
      <td>91</td>
      <td>66</td>
    </tr>
    <tr>
      <th>461</th>
      <td>sunset</td>
      <td>28</td>
      <td>54</td>
    </tr>
    <tr>
      <th>861</th>
      <td>a-kid-like-jake</td>
      <td>52</td>
      <td>60</td>
    </tr>
    <tr>
      <th>767</th>
      <td>eli</td>
      <td>47</td>
      <td>47</td>
    </tr>
    <tr>
      <th>762</th>
      <td>kung-fu-yoga</td>
      <td>47</td>
      <td>49</td>
    </tr>
    <tr>
      <th>168</th>
      <td>all-relative</td>
      <td>9</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1483</th>
      <td>toy-story-4</td>
      <td>96</td>
      <td>40</td>
    </tr>
    <tr>
      <th>561</th>
      <td>6-underground</td>
      <td>35</td>
      <td>48</td>
    </tr>
    <tr>
      <th>568</th>
      <td>honeyglue</td>
      <td>35</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1482</th>
      <td>hate-u-give-the</td>
      <td>96</td>
      <td>42</td>
    </tr>
    <tr>
      <th>237</th>
      <td>vigilante-diaries</td>
      <td>14</td>
      <td>46</td>
    </tr>
    <tr>
      <th>910</th>
      <td>hummingbird-project-the</td>
      <td>56</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>tribe-the</td>
      <td>70</td>
      <td>54</td>
    </tr>
    <tr>
      <th>420</th>
      <td>unlocked</td>
      <td>25</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1333</th>
      <td>12th-man-the</td>
      <td>85</td>
      <td>48</td>
    </tr>
    <tr>
      <th>99</th>
      <td>so-undercover</td>
      <td>5</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1527</th>
      <td>iron-fists-and-kung-fu-kicks</td>
      <td>99</td>
      <td>70</td>
    </tr>
    <tr>
      <th>827</th>
      <td>last-movie-star-the</td>
      <td>51</td>
      <td>40</td>
    </tr>
    <tr>
      <th>415</th>
      <td>killerman</td>
      <td>25</td>
      <td>42</td>
    </tr>
    <tr>
      <th>67</th>
      <td>flatliners</td>
      <td>3</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1109</th>
      <td>kill-team-the</td>
      <td>69</td>
      <td>38</td>
    </tr>
    <tr>
      <th>247</th>
      <td>lets-be-evil</td>
      <td>14</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>moss</td>
      <td>74</td>
      <td>51</td>
    </tr>
    <tr>
      <th>791</th>
      <td>superfly</td>
      <td>49</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1357</th>
      <td>madelines-madeline</td>
      <td>87</td>
      <td>55</td>
    </tr>
    <tr>
      <th>841</th>
      <td>miracle-season-the</td>
      <td>51</td>
      <td>55</td>
    </tr>
    <tr>
      <th>555</th>
      <td>mark-felt-the-man-who-brought-down-the-white-h...</td>
      <td>34</td>
      <td>57</td>
    </tr>
    <tr>
      <th>812</th>
      <td>angelfish</td>
      <td>49</td>
      <td>48</td>
    </tr>
    <tr>
      <th>123</th>
      <td>grown-ups-2</td>
      <td>6</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>ready-player-one</td>
      <td>71</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1388</th>
      <td>souvenir-the</td>
      <td>89</td>
      <td>62</td>
    </tr>
    <tr>
      <th>422</th>
      <td>uglydolls</td>
      <td>26</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>lizzie</td>
      <td>66</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1125</th>
      <td>deadtectives</td>
      <td>70</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>stellas-last-weekend</td>
      <td>64</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>blood-on-her-name</td>
      <td>94</td>
      <td>40</td>
    </tr>
    <tr>
      <th>987</th>
      <td>sicario-day-of-the-soldado</td>
      <td>62</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1264</th>
      <td>daniel-isnt-real</td>
      <td>80</td>
      <td>56</td>
    </tr>
    <tr>
      <th>239</th>
      <td>a-score-to-settle</td>
      <td>14</td>
      <td>39</td>
    </tr>
    <tr>
      <th>367</th>
      <td>1517-to-paris-the</td>
      <td>22</td>
      <td>51</td>
    </tr>
    <tr>
      <th>481</th>
      <td>red-joan</td>
      <td>29</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1525</th>
      <td>after-we-leave</td>
      <td>99</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1404</th>
      <td>luce</td>
      <td>90</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1518</th>
      <td>tower</td>
      <td>98</td>
      <td>66</td>
    </tr>
    <tr>
      <th>993</th>
      <td>marie-curie-the-courage-of-knowledge</td>
      <td>62</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>use-me</td>
      <td>85</td>
      <td>67</td>
    </tr>
    <tr>
      <th>29</th>
      <td>meet-the-spartans</td>
      <td>1</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>suspiria</td>
      <td>64</td>
      <td>53</td>
    </tr>
    <tr>
      <th>342</th>
      <td>book-of-henry-the</td>
      <td>21</td>
      <td>42</td>
    </tr>
    <tr>
      <th>849</th>
      <td>roman-j-israel-esq</td>
      <td>52</td>
      <td>56</td>
    </tr>
    <tr>
      <th>416</th>
      <td>countdown</td>
      <td>25</td>
      <td>38</td>
    </tr>
    <tr>
      <th>65</th>
      <td>a-little-bit-of-heaven</td>
      <td>3</td>
      <td>55</td>
    </tr>
    <tr>
      <th>358</th>
      <td>adderall-diaries-the</td>
      <td>21</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1237</th>
      <td>rondo</td>
      <td>79</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>instant-family</td>
      <td>80</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1038</th>
      <td>little-stranger-the</td>
      <td>65</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>tales-from-the-hood-2</td>
      <td>74</td>
      <td>46</td>
    </tr>
    <tr>
      <th>141</th>
      <td>exposed</td>
      <td>7</td>
      <td>44</td>
    </tr>
    <tr>
      <th>802</th>
      <td>battle-of-jangsari-the</td>
      <td>49</td>
      <td>52</td>
    </tr>
    <tr>
      <th>953</th>
      <td>new-money</td>
      <td>59</td>
      <td>54</td>
    </tr>
    <tr>
      <th>514</th>
      <td>chamber-the</td>
      <td>32</td>
      <td>44</td>
    </tr>
    <tr>
      <th>727</th>
      <td>message-from-the-king</td>
      <td>45</td>
      <td>40</td>
    </tr>
    <tr>
      <th>275</th>
      <td>kissing-booth-the</td>
      <td>16</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1279</th>
      <td>boy-erased</td>
      <td>81</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>roxanne-roxanne</td>
      <td>69</td>
      <td>59</td>
    </tr>
    <tr>
      <th>513</th>
      <td>home-again</td>
      <td>32</td>
      <td>52</td>
    </tr>
    <tr>
      <th>627</th>
      <td>nostalgia</td>
      <td>39</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1142</th>
      <td>a-christmas-prince</td>
      <td>72</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1046</th>
      <td>prophet-the</td>
      <td>65</td>
      <td>50</td>
    </tr>
    <tr>
      <th>928</th>
      <td>liam-gallagher-as-it-was</td>
      <td>58</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>rockaway</td>
      <td>63</td>
      <td>51</td>
    </tr>
    <tr>
      <th>44</th>
      <td>half-past-dead</td>
      <td>2</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>someone-great</td>
      <td>81</td>
      <td>59</td>
    </tr>
    <tr>
      <th>270</th>
      <td>looking-glass</td>
      <td>16</td>
      <td>45</td>
    </tr>
    <tr>
      <th>405</th>
      <td>beast-of-burden</td>
      <td>24</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>oceans-8</td>
      <td>68</td>
      <td>57</td>
    </tr>
    <tr>
      <th>115</th>
      <td>paranoia</td>
      <td>6</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>white-crow-the</td>
      <td>66</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>i-think-were-alone-now</td>
      <td>62</td>
      <td>42</td>
    </tr>
    <tr>
      <th>795</th>
      <td>cuck</td>
      <td>49</td>
      <td>42</td>
    </tr>
    <tr>
      <th>528</th>
      <td>an-actor-prepares</td>
      <td>32</td>
      <td>61</td>
    </tr>
    <tr>
      <th>432</th>
      <td>sherlock-gnomes</td>
      <td>26</td>
      <td>47</td>
    </tr>
    <tr>
      <th>896</th>
      <td>beach-bum-the</td>
      <td>55</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1200</th>
      <td>braven</td>
      <td>76</td>
      <td>44</td>
    </tr>
    <tr>
      <th>78</th>
      <td>hangman</td>
      <td>4</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>anthony-jeselnik-fire-in-the-maternity-ward</td>
      <td>79</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1534</th>
      <td>mike-birbiglia-the-new-one</td>
      <td>99</td>
      <td>55</td>
    </tr>
    <tr>
      <th>939</th>
      <td>line-of-duty</td>
      <td>58</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1084</th>
      <td>journey-the</td>
      <td>68</td>
      <td>54</td>
    </tr>
    <tr>
      <th>630</th>
      <td>gringo</td>
      <td>39</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1045</th>
      <td>polka-king-the</td>
      <td>65</td>
      <td>47</td>
    </tr>
    <tr>
      <th>15</th>
      <td>dead-water</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1121</th>
      <td>black-site</td>
      <td>70</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1030</th>
      <td>resistance</td>
      <td>64</td>
      <td>62</td>
    </tr>
    <tr>
      <th>175</th>
      <td>last-heist-the</td>
      <td>10</td>
      <td>39</td>
    </tr>
    <tr>
      <th>989</th>
      <td>oath-the</td>
      <td>62</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>vanishing-the</td>
      <td>84</td>
      <td>42</td>
    </tr>
    <tr>
      <th>868</th>
      <td>crash-pad</td>
      <td>53</td>
      <td>29</td>
    </tr>
    <tr>
      <th>506</th>
      <td>intruder-the</td>
      <td>31</td>
      <td>36</td>
    </tr>
    <tr>
      <th>943</th>
      <td>ophelia</td>
      <td>59</td>
      <td>55</td>
    </tr>
    <tr>
      <th>365</th>
      <td>bad-santa-2</td>
      <td>22</td>
      <td>45</td>
    </tr>
    <tr>
      <th>549</th>
      <td>live-by-night</td>
      <td>34</td>
      <td>35</td>
    </tr>
    <tr>
      <th>203</th>
      <td>viking-destiny</td>
      <td>12</td>
      <td>50</td>
    </tr>
    <tr>
      <th>783</th>
      <td>churchill</td>
      <td>48</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1225</th>
      <td>dark-river</td>
      <td>78</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>creed-ii</td>
      <td>83</td>
      <td>61</td>
    </tr>
    <tr>
      <th>1266</th>
      <td>judy</td>
      <td>81</td>
      <td>25</td>
    </tr>
    <tr>
      <th>811</th>
      <td>pimp</td>
      <td>49</td>
      <td>45</td>
    </tr>
    <tr>
      <th>913</th>
      <td>ray-romano-right-here-around-the-corner</td>
      <td>56</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1392</th>
      <td>lighthouse-the</td>
      <td>89</td>
      <td>53</td>
    </tr>
    <tr>
      <th>184</th>
      <td>ghost-team</td>
      <td>10</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>bikram-yogi-guru-predator</td>
      <td>95</td>
      <td>69</td>
    </tr>
    <tr>
      <th>479</th>
      <td>king-arthur-legend-of-the-sword</td>
      <td>29</td>
      <td>49</td>
    </tr>
    <tr>
      <th>56</th>
      <td>down-to-you</td>
      <td>2</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>batman-gotham-by-gaslight</td>
      <td>72</td>
      <td>56</td>
    </tr>
    <tr>
      <th>486</th>
      <td>rodin</td>
      <td>29</td>
      <td>48</td>
    </tr>
    <tr>
      <th>383</th>
      <td>kill-your-friends</td>
      <td>23</td>
      <td>57</td>
    </tr>
    <tr>
      <th>43</th>
      <td>passion-play</td>
      <td>2</td>
      <td>43</td>
    </tr>
    <tr>
      <th>533</th>
      <td>poms</td>
      <td>33</td>
      <td>59</td>
    </tr>
    <tr>
      <th>758</th>
      <td>going-in-style</td>
      <td>47</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>wonders-of-the-sea</td>
      <td>75</td>
      <td>44</td>
    </tr>
    <tr>
      <th>597</th>
      <td>i-am-vengeance</td>
      <td>37</td>
      <td>33</td>
    </tr>
    <tr>
      <th>101</th>
      <td>crucifixion-the</td>
      <td>5</td>
      <td>54</td>
    </tr>
    <tr>
      <th>552</th>
      <td>welcome-to-marwen</td>
      <td>34</td>
      <td>40</td>
    </tr>
    <tr>
      <th>579</th>
      <td>wetlands</td>
      <td>35</td>
      <td>47</td>
    </tr>
    <tr>
      <th>107</th>
      <td>fifty-shades-of-black</td>
      <td>6</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1294</th>
      <td>coyote-lake</td>
      <td>82</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>zombieland-double-tap</td>
      <td>67</td>
      <td>43</td>
    </tr>
    <tr>
      <th>220</th>
      <td>when-the-bough-breaks</td>
      <td>12</td>
      <td>28</td>
    </tr>
    <tr>
      <th>946</th>
      <td>operation-finale</td>
      <td>59</td>
      <td>36</td>
    </tr>
    <tr>
      <th>451</th>
      <td>batman-v-superman-dawn-of-justice</td>
      <td>27</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1502</th>
      <td>moonlight</td>
      <td>98</td>
      <td>64</td>
    </tr>
    <tr>
      <th>889</th>
      <td>commuter-the</td>
      <td>55</td>
      <td>35</td>
    </tr>
    <tr>
      <th>725</th>
      <td>paul-apostle-of-christ</td>
      <td>45</td>
      <td>48</td>
    </tr>
    <tr>
      <th>128</th>
      <td>always-woodstock</td>
      <td>7</td>
      <td>52</td>
    </tr>
    <tr>
      <th>332</th>
      <td>terminal</td>
      <td>19</td>
      <td>38</td>
    </tr>
    <tr>
      <th>297</th>
      <td>chips</td>
      <td>18</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1466</th>
      <td>blackkklansman</td>
      <td>95</td>
      <td>26</td>
    </tr>
    <tr>
      <th>706</th>
      <td>jackals</td>
      <td>44</td>
      <td>45</td>
    </tr>
    <tr>
      <th>680</th>
      <td>15-minutes-of-war</td>
      <td>42</td>
      <td>51</td>
    </tr>
    <tr>
      <th>244</th>
      <td>transformers-the-last-knight</td>
      <td>14</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1522</th>
      <td>parting-glass-the</td>
      <td>99</td>
      <td>41</td>
    </tr>
    <tr>
      <th>662</th>
      <td>valley-the</td>
      <td>41</td>
      <td>50</td>
    </tr>
    <tr>
      <th>847</th>
      <td>lion-king-the</td>
      <td>52</td>
      <td>36</td>
    </tr>
    <tr>
      <th>708</th>
      <td>girls-of-the-sun</td>
      <td>44</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1171</th>
      <td>after-louie</td>
      <td>74</td>
      <td>44</td>
    </tr>
    <tr>
      <th>231</th>
      <td>circle-the</td>
      <td>13</td>
      <td>42</td>
    </tr>
    <tr>
      <th>32</th>
      <td>daddy-day-camp</td>
      <td>1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1201</th>
      <td>i-kill-giants</td>
      <td>76</td>
      <td>50</td>
    </tr>
    <tr>
      <th>494</th>
      <td>mapplethorpe</td>
      <td>30</td>
      <td>52</td>
    </tr>
    <tr>
      <th>599</th>
      <td>kidnap</td>
      <td>37</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>who-would-you-take-to-a-deserted-island</td>
      <td>82</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1185</th>
      <td>mirage</td>
      <td>74</td>
      <td>59</td>
    </tr>
    <tr>
      <th>350</th>
      <td>6-below-miracle-on-the-mountain</td>
      <td>21</td>
      <td>43</td>
    </tr>
    <tr>
      <th>992</th>
      <td>childs-play</td>
      <td>62</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1155</th>
      <td>on-the-basis-of-sex</td>
      <td>72</td>
      <td>41</td>
    </tr>
    <tr>
      <th>967</th>
      <td>hotel-transylvania-3-summer-vacation</td>
      <td>61</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>dumplin</td>
      <td>84</td>
      <td>43</td>
    </tr>
    <tr>
      <th>411</th>
      <td>unforgettable</td>
      <td>25</td>
      <td>48</td>
    </tr>
    <tr>
      <th>163</th>
      <td>stonewall</td>
      <td>9</td>
      <td>49</td>
    </tr>
    <tr>
      <th>906</th>
      <td>154</td>
      <td>56</td>
      <td>56</td>
    </tr>
    <tr>
      <th>497</th>
      <td>wonder-wheel</td>
      <td>30</td>
      <td>40</td>
    </tr>
    <tr>
      <th>306</th>
      <td>possession-of-hannah-grace-the</td>
      <td>18</td>
      <td>50</td>
    </tr>
    <tr>
      <th>433</th>
      <td>escape-plan-the-extractors</td>
      <td>26</td>
      <td>29</td>
    </tr>
    <tr>
      <th>671</th>
      <td>katie-says-goodbye</td>
      <td>41</td>
      <td>35</td>
    </tr>
    <tr>
      <th>600</th>
      <td>el-chicano</td>
      <td>37</td>
      <td>40</td>
    </tr>
    <tr>
      <th>371</th>
      <td>rim-of-the-world</td>
      <td>22</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1426</th>
      <td>fighting-with-my-family</td>
      <td>92</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>speed-of-life</td>
      <td>77</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1088</th>
      <td>final-score</td>
      <td>68</td>
      <td>56</td>
    </tr>
    <tr>
      <th>192</th>
      <td>basement-the</td>
      <td>10</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>triple-frontier</td>
      <td>70</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>terminator-dark-fate</td>
      <td>69</td>
      <td>40</td>
    </tr>
    <tr>
      <th>929</th>
      <td>astronaut</td>
      <td>58</td>
      <td>43</td>
    </tr>
    <tr>
      <th>63</th>
      <td>playing-for-keeps</td>
      <td>3</td>
      <td>46</td>
    </tr>
    <tr>
      <th>709</th>
      <td>demon-house</td>
      <td>44</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1150</th>
      <td>galveston</td>
      <td>72</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1207</th>
      <td>thank-you-for-your-service</td>
      <td>76</td>
      <td>32</td>
    </tr>
    <tr>
      <th>942</th>
      <td>bad-moms</td>
      <td>58</td>
      <td>54</td>
    </tr>
    <tr>
      <th>911</th>
      <td>lbj</td>
      <td>56</td>
      <td>50</td>
    </tr>
    <tr>
      <th>274</th>
      <td>look-away</td>
      <td>16</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>prospect</td>
      <td>88</td>
      <td>49</td>
    </tr>
    <tr>
      <th>670</th>
      <td>black-butterfly</td>
      <td>41</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1382</th>
      <td>blinded-by-the-light</td>
      <td>88</td>
      <td>55</td>
    </tr>
    <tr>
      <th>198</th>
      <td>fifty-shades-freed</td>
      <td>11</td>
      <td>29</td>
    </tr>
    <tr>
      <th>453</th>
      <td>into-the-ashes</td>
      <td>27</td>
      <td>52</td>
    </tr>
    <tr>
      <th>339</th>
      <td>phil</td>
      <td>20</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1039</th>
      <td>danger-close</td>
      <td>65</td>
      <td>48</td>
    </tr>
    <tr>
      <th>892</th>
      <td>greatest-showman-the</td>
      <td>55</td>
      <td>44</td>
    </tr>
    <tr>
      <th>271</th>
      <td>last-summer-the</td>
      <td>16</td>
      <td>52</td>
    </tr>
    <tr>
      <th>577</th>
      <td>most-hated-woman-in-america-the</td>
      <td>35</td>
      <td>38</td>
    </tr>
    <tr>
      <th>218</th>
      <td>day-of-the-dead-bloodline</td>
      <td>12</td>
      <td>36</td>
    </tr>
    <tr>
      <th>799</th>
      <td>postcards-from-london</td>
      <td>49</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1241</th>
      <td>greener-grass</td>
      <td>79</td>
      <td>50</td>
    </tr>
    <tr>
      <th>427</th>
      <td>week-of-the</td>
      <td>26</td>
      <td>51</td>
    </tr>
    <tr>
      <th>351</th>
      <td>i-hate-kids</td>
      <td>21</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>before-i-fall</td>
      <td>64</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1267</th>
      <td>room-for-rent</td>
      <td>81</td>
      <td>54</td>
    </tr>
    <tr>
      <th>324</th>
      <td>beneath-the-leaves</td>
      <td>19</td>
      <td>49</td>
    </tr>
    <tr>
      <th>413</th>
      <td>undrafted</td>
      <td>25</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>what-they-had</td>
      <td>86</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1043</th>
      <td>body-at-brighton-rock</td>
      <td>65</td>
      <td>60</td>
    </tr>
    <tr>
      <th>696</th>
      <td>ready-to-mingle</td>
      <td>43</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>wrinkles-the-clown</td>
      <td>69</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>hale-county-this-morning-this-evening</td>
      <td>96</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1159</th>
      <td>ghost-light</td>
      <td>73</td>
      <td>57</td>
    </tr>
    <tr>
      <th>289</th>
      <td>great-war-the</td>
      <td>16</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1173</th>
      <td>anchor-and-hope</td>
      <td>74</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1427</th>
      <td>last-race-the</td>
      <td>92</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>tribes-of-palos-verdes-the</td>
      <td>63</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>unicorn-store</td>
      <td>63</td>
      <td>62</td>
    </tr>
    <tr>
      <th>619</th>
      <td>slaughterhouse-rulez</td>
      <td>38</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>changeland</td>
      <td>62</td>
      <td>34</td>
    </tr>
    <tr>
      <th>277</th>
      <td>jeepers-creepers-3</td>
      <td>16</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>mega-time-squad</td>
      <td>75</td>
      <td>53</td>
    </tr>
    <tr>
      <th>864</th>
      <td>47-meters-down</td>
      <td>53</td>
      <td>56</td>
    </tr>
    <tr>
      <th>236</th>
      <td>open-house-the</td>
      <td>14</td>
      <td>58</td>
    </tr>
    <tr>
      <th>615</th>
      <td>whisky-galore</td>
      <td>38</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>bill-burr-paper-tiger</td>
      <td>85</td>
      <td>66</td>
    </tr>
    <tr>
      <th>1480</th>
      <td>gangster-the-cop-the-devil-the</td>
      <td>96</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1402</th>
      <td>mirai</td>
      <td>90</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>downton-abbey</td>
      <td>83</td>
      <td>51</td>
    </tr>
    <tr>
      <th>208</th>
      <td>sword-of-vengeance</td>
      <td>12</td>
      <td>58</td>
    </tr>
    <tr>
      <th>380</th>
      <td>mr-church</td>
      <td>23</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1252</th>
      <td>kevin-harts-guide-to-black-history</td>
      <td>79</td>
      <td>52</td>
    </tr>
    <tr>
      <th>941</th>
      <td>absolutely-fabulous-the-movie</td>
      <td>58</td>
      <td>39</td>
    </tr>
    <tr>
      <th>58</th>
      <td>happily-never-after</td>
      <td>3</td>
      <td>64</td>
    </tr>
    <tr>
      <th>382</th>
      <td>90-minutes-in-heaven</td>
      <td>23</td>
      <td>48</td>
    </tr>
    <tr>
      <th>374</th>
      <td>clapper-the</td>
      <td>22</td>
      <td>41</td>
    </tr>
    <tr>
      <th>429</th>
      <td>rambo-last-blood</td>
      <td>26</td>
      <td>55</td>
    </tr>
    <tr>
      <th>997</th>
      <td>it-chapter-two</td>
      <td>62</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1230</th>
      <td>buffalo-boys</td>
      <td>78</td>
      <td>34</td>
    </tr>
    <tr>
      <th>744</th>
      <td>hollars-the</td>
      <td>46</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1240</th>
      <td>30-miles-from-nowhere</td>
      <td>79</td>
      <td>32</td>
    </tr>
    <tr>
      <th>361</th>
      <td>ithaca</td>
      <td>21</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1083</th>
      <td>disappearance-at-clifton-hill</td>
      <td>68</td>
      <td>53</td>
    </tr>
    <tr>
      <th>464</th>
      <td>new-life</td>
      <td>28</td>
      <td>56</td>
    </tr>
    <tr>
      <th>583</th>
      <td>resident-evil-the-final-chapter</td>
      <td>36</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>spider-in-the-web</td>
      <td>74</td>
      <td>42</td>
    </tr>
    <tr>
      <th>844</th>
      <td>magic-in-the-moonlight</td>
      <td>51</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>adopt-a-highway</td>
      <td>67</td>
      <td>52</td>
    </tr>
    <tr>
      <th>233</th>
      <td>nine-lives</td>
      <td>14</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>mustang-the</td>
      <td>93</td>
      <td>61</td>
    </tr>
    <tr>
      <th>196</th>
      <td>point-break</td>
      <td>11</td>
      <td>50</td>
    </tr>
    <tr>
      <th>665</th>
      <td>aftermath</td>
      <td>41</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>afterward</td>
      <td>82</td>
      <td>21</td>
    </tr>
    <tr>
      <th>81</th>
      <td>abduction</td>
      <td>4</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1438</th>
      <td>screwball</td>
      <td>93</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>dog-days</td>
      <td>62</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>favourite-the</td>
      <td>92</td>
      <td>56</td>
    </tr>
    <tr>
      <th>111</th>
      <td>cbgb</td>
      <td>6</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>rust-creek</td>
      <td>83</td>
      <td>35</td>
    </tr>
    <tr>
      <th>809</th>
      <td>a-christmas-prince-the-royal-wedding</td>
      <td>49</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>day-shall-come-the</td>
      <td>64</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>avengers-endgame</td>
      <td>93</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1298</th>
      <td>official-secrets</td>
      <td>82</td>
      <td>41</td>
    </tr>
    <tr>
      <th>676</th>
      <td>ghost-stories</td>
      <td>42</td>
      <td>59</td>
    </tr>
    <tr>
      <th>310</th>
      <td>diary-of-a-wimpy-kid-the-long-haul</td>
      <td>18</td>
      <td>47</td>
    </tr>
    <tr>
      <th>612</th>
      <td>social-animals</td>
      <td>37</td>
      <td>53</td>
    </tr>
    <tr>
      <th>925</th>
      <td>close-enemies</td>
      <td>57</td>
      <td>35</td>
    </tr>
    <tr>
      <th>920</th>
      <td>hippopotamus-the</td>
      <td>57</td>
      <td>53</td>
    </tr>
    <tr>
      <th>848</th>
      <td>american-dreamer</td>
      <td>52</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>snapshots</td>
      <td>71</td>
      <td>51</td>
    </tr>
    <tr>
      <th>620</th>
      <td>death-note</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>366</th>
      <td>mile-22</td>
      <td>22</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1523</th>
      <td>toxic-beauty</td>
      <td>99</td>
      <td>68</td>
    </tr>
    <tr>
      <th>1247</th>
      <td>fire-in-paradise</td>
      <td>79</td>
      <td>55</td>
    </tr>
    <tr>
      <th>425</th>
      <td>victor-frankenstein</td>
      <td>26</td>
      <td>41</td>
    </tr>
    <tr>
      <th>884</th>
      <td>mrs-hyde</td>
      <td>54</td>
      <td>52</td>
    </tr>
    <tr>
      <th>832</th>
      <td>boss-baby-the</td>
      <td>51</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>a-private-war</td>
      <td>87</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1408</th>
      <td>how-to-train-your-dragon-the-hidden-world</td>
      <td>90</td>
      <td>42</td>
    </tr>
    <tr>
      <th>650</th>
      <td>spinning-man</td>
      <td>41</td>
      <td>45</td>
    </tr>
    <tr>
      <th>589</th>
      <td>leisure-seeker-the</td>
      <td>36</td>
      <td>57</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>last-flag-flying</td>
      <td>77</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>sword-of-trust</td>
      <td>92</td>
      <td>43</td>
    </tr>
    <tr>
      <th>785</th>
      <td>freeheld</td>
      <td>48</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>lego-movie-2-the-second-part-the</td>
      <td>84</td>
      <td>53</td>
    </tr>
    <tr>
      <th>254</th>
      <td>sinister-2</td>
      <td>15</td>
      <td>26</td>
    </tr>
    <tr>
      <th>322</th>
      <td>hospitality</td>
      <td>19</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1118</th>
      <td>horse-girl</td>
      <td>70</td>
      <td>54</td>
    </tr>
    <tr>
      <th>428</th>
      <td>little-women</td>
      <td>26</td>
      <td>79</td>
    </tr>
    <tr>
      <th>576</th>
      <td>i-feel-pretty</td>
      <td>35</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1078</th>
      <td>puppet-master-the-littlest-reich</td>
      <td>67</td>
      <td>48</td>
    </tr>
    <tr>
      <th>872</th>
      <td>gemma-bovery</td>
      <td>53</td>
      <td>44</td>
    </tr>
    <tr>
      <th>354</th>
      <td>zoolander-2</td>
      <td>21</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>invader-zim-enter-the-florpus</td>
      <td>99</td>
      <td>52</td>
    </tr>
    <tr>
      <th>86</th>
      <td>i-frankenstein</td>
      <td>4</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>burning</td>
      <td>94</td>
      <td>52</td>
    </tr>
    <tr>
      <th>265</th>
      <td>mummy-the</td>
      <td>15</td>
      <td>28</td>
    </tr>
    <tr>
      <th>287</th>
      <td>dear-dictator</td>
      <td>16</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1519</th>
      <td>things-to-come</td>
      <td>98</td>
      <td>62</td>
    </tr>
    <tr>
      <th>438</th>
      <td>drone</td>
      <td>26</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>premature</td>
      <td>92</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1089</th>
      <td>housemaid-the</td>
      <td>68</td>
      <td>45</td>
    </tr>
    <tr>
      <th>439</th>
      <td>duel-the</td>
      <td>26</td>
      <td>45</td>
    </tr>
    <tr>
      <th>759</th>
      <td>war-machine</td>
      <td>47</td>
      <td>45</td>
    </tr>
    <tr>
      <th>221</th>
      <td>american-heist</td>
      <td>12</td>
      <td>53</td>
    </tr>
    <tr>
      <th>113</th>
      <td>java-heat</td>
      <td>6</td>
      <td>33</td>
    </tr>
    <tr>
      <th>591</th>
      <td>in-the-tall-grass</td>
      <td>36</td>
      <td>40</td>
    </tr>
    <tr>
      <th>602</th>
      <td>backstabbing-for-beginners</td>
      <td>37</td>
      <td>59</td>
    </tr>
    <tr>
      <th>174</th>
      <td>vanishing-of-sidney-hall-the</td>
      <td>10</td>
      <td>40</td>
    </tr>
    <tr>
      <th>553</th>
      <td>vhs-viral</td>
      <td>34</td>
      <td>51</td>
    </tr>
    <tr>
      <th>381</th>
      <td>nobodys-fool</td>
      <td>23</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>standoff-at-sparrow-creek-the</td>
      <td>76</td>
      <td>61</td>
    </tr>
    <tr>
      <th>18</th>
      <td>london-fields</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1194</th>
      <td>in-the-fade</td>
      <td>76</td>
      <td>50</td>
    </tr>
    <tr>
      <th>285</th>
      <td>competition-the</td>
      <td>16</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>black-47</td>
      <td>77</td>
      <td>66</td>
    </tr>
    <tr>
      <th>749</th>
      <td>clown</td>
      <td>46</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>revival-the</td>
      <td>78</td>
      <td>47</td>
    </tr>
    <tr>
      <th>31</th>
      <td>baby-geniuses</td>
      <td>1</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1165</th>
      <td>claras-ghost</td>
      <td>73</td>
      <td>54</td>
    </tr>
    <tr>
      <th>210</th>
      <td>mortdecai</td>
      <td>12</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>lasso</td>
      <td>62</td>
      <td>43</td>
    </tr>
    <tr>
      <th>478</th>
      <td>outlaws</td>
      <td>29</td>
      <td>56</td>
    </tr>
    <tr>
      <th>839</th>
      <td>rampage</td>
      <td>51</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1531</th>
      <td>2-in-the-bush-a-love-story</td>
      <td>99</td>
      <td>50</td>
    </tr>
    <tr>
      <th>965</th>
      <td>alita-battle-angel</td>
      <td>60</td>
      <td>32</td>
    </tr>
    <tr>
      <th>394</th>
      <td>rememory</td>
      <td>24</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>una</td>
      <td>76</td>
      <td>51</td>
    </tr>
    <tr>
      <th>188</th>
      <td>great-alaskan-race-the</td>
      <td>10</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>angel-the</td>
      <td>74</td>
      <td>48</td>
    </tr>
    <tr>
      <th>969</th>
      <td>mary-queen-of-scots</td>
      <td>61</td>
      <td>51</td>
    </tr>
    <tr>
      <th>635</th>
      <td>last-witness-the</td>
      <td>39</td>
      <td>46</td>
    </tr>
    <tr>
      <th>493</th>
      <td>loving-pablo</td>
      <td>30</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>hustlers</td>
      <td>86</td>
      <td>57</td>
    </tr>
    <tr>
      <th>934</th>
      <td>white-boy-rick</td>
      <td>58</td>
      <td>42</td>
    </tr>
    <tr>
      <th>83</th>
      <td>home-sweet-hell</td>
      <td>4</td>
      <td>28</td>
    </tr>
    <tr>
      <th>695</th>
      <td>wedding-guest-the</td>
      <td>43</td>
      <td>34</td>
    </tr>
    <tr>
      <th>48</th>
      <td>bless-the-child</td>
      <td>2</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>long-dumb-road-the</td>
      <td>72</td>
      <td>49</td>
    </tr>
    <tr>
      <th>155</th>
      <td>dont-sleep</td>
      <td>8</td>
      <td>47</td>
    </tr>
    <tr>
      <th>426</th>
      <td>london-has-fallen</td>
      <td>26</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1167</th>
      <td>sobibor</td>
      <td>74</td>
      <td>52</td>
    </tr>
    <tr>
      <th>435</th>
      <td>mortal-engines</td>
      <td>26</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1076</th>
      <td>joker</td>
      <td>67</td>
      <td>50</td>
    </tr>
    <tr>
      <th>837</th>
      <td>tomb-raider</td>
      <td>51</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>vfw</td>
      <td>80</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1516</th>
      <td>gods-own-country</td>
      <td>98</td>
      <td>53</td>
    </tr>
    <tr>
      <th>536</th>
      <td>wonder-park</td>
      <td>33</td>
      <td>40</td>
    </tr>
    <tr>
      <th>179</th>
      <td>i-am-wrath</td>
      <td>10</td>
      <td>47</td>
    </tr>
    <tr>
      <th>544</th>
      <td>abattoir</td>
      <td>34</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1278</th>
      <td>level-16</td>
      <td>81</td>
      <td>62</td>
    </tr>
    <tr>
      <th>632</th>
      <td>run-the-race</td>
      <td>39</td>
      <td>44</td>
    </tr>
    <tr>
      <th>717</th>
      <td>everything-everything</td>
      <td>45</td>
      <td>53</td>
    </tr>
    <tr>
      <th>846</th>
      <td>genius</td>
      <td>52</td>
      <td>48</td>
    </tr>
    <tr>
      <th>363</th>
      <td>overdrive</td>
      <td>22</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1205</th>
      <td>jumanji-welcome-to-the-jungle</td>
      <td>76</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1228</th>
      <td>all-the-money-in-the-world</td>
      <td>78</td>
      <td>43</td>
    </tr>
    <tr>
      <th>346</th>
      <td>14-cameras</td>
      <td>21</td>
      <td>53</td>
    </tr>
    <tr>
      <th>821</th>
      <td>miss-julie</td>
      <td>51</td>
      <td>52</td>
    </tr>
    <tr>
      <th>535</th>
      <td>fun-mom-dinner</td>
      <td>33</td>
      <td>62</td>
    </tr>
    <tr>
      <th>681</th>
      <td>maze-runner-the-death-cure</td>
      <td>42</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1325</th>
      <td>untouchable</td>
      <td>84</td>
      <td>55</td>
    </tr>
    <tr>
      <th>608</th>
      <td>along-came-the-devil</td>
      <td>37</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1274</th>
      <td>ben-is-back</td>
      <td>81</td>
      <td>55</td>
    </tr>
    <tr>
      <th>944</th>
      <td>scarborough</td>
      <td>59</td>
      <td>52</td>
    </tr>
    <tr>
      <th>243</th>
      <td>sharknado-the-4th-awakens</td>
      <td>14</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1491</th>
      <td>homecoming-a-film-by-beyonce</td>
      <td>97</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1430</th>
      <td>high-flying-bird</td>
      <td>92</td>
      <td>58</td>
    </tr>
    <tr>
      <th>637</th>
      <td>girl-in-the-spiders-web-the</td>
      <td>39</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>wildling</td>
      <td>71</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>all-the-bright-places</td>
      <td>75</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>68-kill</td>
      <td>68</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>private-life</td>
      <td>94</td>
      <td>46</td>
    </tr>
    <tr>
      <th>815</th>
      <td>an-evening-with-beverly-luff-linn</td>
      <td>50</td>
      <td>44</td>
    </tr>
    <tr>
      <th>308</th>
      <td>underworld-blood-wars</td>
      <td>18</td>
      <td>39</td>
    </tr>
    <tr>
      <th>430</th>
      <td>can-you-keep-a-secret</td>
      <td>26</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>solo-a-star-wars-story</td>
      <td>69</td>
      <td>37</td>
    </tr>
    <tr>
      <th>398</th>
      <td>otherhood</td>
      <td>24</td>
      <td>47</td>
    </tr>
    <tr>
      <th>423</th>
      <td>hellions</td>
      <td>26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>370</th>
      <td>dark-phoenix</td>
      <td>22</td>
      <td>49</td>
    </tr>
    <tr>
      <th>54</th>
      <td>deuces-wild</td>
      <td>2</td>
      <td>40</td>
    </tr>
    <tr>
      <th>790</th>
      <td>sonata-the</td>
      <td>49</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1169</th>
      <td>monster-party</td>
      <td>74</td>
      <td>40</td>
    </tr>
    <tr>
      <th>996</th>
      <td>bird-box</td>
      <td>62</td>
      <td>43</td>
    </tr>
    <tr>
      <th>312</th>
      <td>blind</td>
      <td>18</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>first-man</td>
      <td>86</td>
      <td>58</td>
    </tr>
    <tr>
      <th>592</th>
      <td>morgan</td>
      <td>36</td>
      <td>44</td>
    </tr>
    <tr>
      <th>109</th>
      <td>amityville-murders-the</td>
      <td>6</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>changeover-the</td>
      <td>68</td>
      <td>43</td>
    </tr>
    <tr>
      <th>813</th>
      <td>escape-room</td>
      <td>49</td>
      <td>36</td>
    </tr>
    <tr>
      <th>100</th>
      <td>extraction</td>
      <td>5</td>
      <td>38</td>
    </tr>
    <tr>
      <th>482</th>
      <td>leatherface</td>
      <td>29</td>
      <td>32</td>
    </tr>
    <tr>
      <th>693</th>
      <td>tall-girl</td>
      <td>43</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1409</th>
      <td>rolling-thunder-revue-a-bob-dylan-story-by-mar...</td>
      <td>91</td>
      <td>76</td>
    </tr>
    <tr>
      <th>266</th>
      <td>transporter-refueled-the</td>
      <td>15</td>
      <td>28</td>
    </tr>
    <tr>
      <th>702</th>
      <td>red-sparrow</td>
      <td>44</td>
      <td>35</td>
    </tr>
    <tr>
      <th>296</th>
      <td>god-bless-the-broken-road</td>
      <td>17</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>colette</td>
      <td>86</td>
      <td>53</td>
    </tr>
    <tr>
      <th>977</th>
      <td>uncle-drew</td>
      <td>61</td>
      <td>43</td>
    </tr>
    <tr>
      <th>723</th>
      <td>meg-the</td>
      <td>45</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1208</th>
      <td>little-pink-house</td>
      <td>76</td>
      <td>57</td>
    </tr>
    <tr>
      <th>377</th>
      <td>birth-of-the-dragon</td>
      <td>23</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
</div>



### Features Only with XG Boost Regressor

Setting up all the features for a features-only (non-tfidf) regression model.


```python
# Need to change this as the model chokes on the period.
rotten_model.rename(columns={'.':'PER'}, inplace=True)

# Dropping out the unnecessary columns.
to_drop = list(rotten_model.columns[:7])
to_drop.append(rotten_model.columns[9])
to_drop.extend(rotten_model.columns[19:22])
to_drop.extend(rotten_model.columns[-2:])

to_drop_2 = ['sentiment_compound', 'unique_words', 'X', 'JJ', 'NNP', 'DET',
             'semi_ratios', 'ADJ']

predictors_all = rotten_df_cut.copy()
predictors_all.drop(to_drop, axis=1, inplace=True)
predictors_all.drop(to_drop_2, axis=1,inplace=True)

# Adding dummy categories.
# rotten_dummies = pd.get_dummies(rotten_model['category'])
# predictors_all = pd.merge(predictors_all, rotten_dummies, left_index=True, 
#              right_index=True)
```


```python
X_train, X_test, y_train, y_test = train_test_split(predictors_all,
                                                    rotten_model.rotten_scores,
                                                    test_size=.3)
```


```python
scaler = StandardScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                          columns = X_train.columns)
scaled_X_test = pd.DataFrame(scaler.transform(X_test), 
                            columns=X_test.columns)
```


```python
scaled_y_train = scaler.fit_transform(np.array(y_train).reshape(-1,1))
scaled_y_test = scaler.transform(np.array(y_test).reshape(-1,1))
scaled_y_train = pd.Series(scaled_y_train.reshape(-1,), name='rotten_scores')
scaled_y_test = pd.Series(scaled_y_test.reshape(-1,), name='rotten_scores')
```


```python
scaled_X_train.fillna(0, inplace=True)
scaled_X_test.fillna(0, inplace=True)
```


```python
def test(x):
    if x == 0:
        return math.log((X + 1) / 100)
    else:
        return math.log(x / 100)
```


```python
model = XGBRegressor(random_state=42, n_estimators=100, 
                     objective='reg:squarederror')
model.fit(scaled_X_train, y_train)
```

    //anaconda3/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning:
    
    Series.base is deprecated and will be removed in a future version
    





    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0,
                 importance_type='gain', learning_rate=0.1, max_delta_step=0,
                 max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:squarederror',
                 random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 seed=None, silent=None, subsample=1, verbosity=1)



Making predictions and calculating residuals


```python
y_hat_train = model.predict(pd.DataFrame(scaled_X_train))
y_hat_test = model.predict(pd.DataFrame(scaled_X_test))

train_residuals = y_hat_train - list(y_train)
test_residuals = y_hat_test - list(y_test)
```


```python
print('Ave deviation from actual: ', round(sum(abs(test_residuals)) / len(test_residuals),2))
```

    Ave deviation from actual:  23.25



```python
mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test = np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)
```

    Train Mean Squarred Error: 339.5014011625739
    Test Mean Squarred Error: 775.9880641716477



```python
print('R^2 score:', r2_score(y_test,y_hat_test))
```

    R^2 score: 0.06923286403294127



```python
y_pred = model.predict(scaled_X_test)
```

Creating dataframe of predicted scores vs. actual


```python
actual_v_predicted = X_test.copy()

pred_scores = list(y_pred)

actual_v_predicted['predicted_scores'] = pred_scores

to_merge = rotten_model[['titles', 'rotten_scores']]

actual_v_predicted = actual_v_predicted.merge(to_merge, left_index=True,
                                              right_index=True)

actual_v_predicted = actual_v_predicted[['titles','rotten_scores',
                                         'predicted_scores']]

actual_v_predicted['predicted_scores'] = actual_v_predicted.predicted_scores.\
    apply(lambda x: int(x))

actual_v_predicted = actual_v_predicted[actual_v_predicted['predicted_scores'] <= 100]
```


```python
actual_v_predicted.predicted_scores.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x53ebc4588>




![png](main_files/main_239_1.png)



```python
actual_v_predicted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>titles</th>
      <th>rotten_scores</th>
      <th>predicted_scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>459</th>
      <td>bethany</td>
      <td>28</td>
      <td>41</td>
    </tr>
    <tr>
      <th>224</th>
      <td>sextuplets</td>
      <td>13</td>
      <td>39</td>
    </tr>
    <tr>
      <th>678</th>
      <td>what-men-want</td>
      <td>42</td>
      <td>49</td>
    </tr>
    <tr>
      <th>174</th>
      <td>vanishing-of-sidney-hall-the</td>
      <td>10</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1164</th>
      <td>mercy-the</td>
      <td>73</td>
      <td>50</td>
    </tr>
    <tr>
      <th>468</th>
      <td>shock-and-awe</td>
      <td>28</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>paterno</td>
      <td>69</td>
      <td>52</td>
    </tr>
    <tr>
      <th>138</th>
      <td>rings</td>
      <td>7</td>
      <td>45</td>
    </tr>
    <tr>
      <th>590</th>
      <td>close</td>
      <td>36</td>
      <td>43</td>
    </tr>
    <tr>
      <th>245</th>
      <td>regression</td>
      <td>14</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1512</th>
      <td>amazing-grace</td>
      <td>98</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1238</th>
      <td>anthony-jeselnik-fire-in-the-maternity-ward</td>
      <td>79</td>
      <td>67</td>
    </tr>
    <tr>
      <th>478</th>
      <td>outlaws</td>
      <td>29</td>
      <td>58</td>
    </tr>
    <tr>
      <th>13</th>
      <td>imprisoned</td>
      <td>0</td>
      <td>61</td>
    </tr>
    <tr>
      <th>898</th>
      <td>cannibal-club-the</td>
      <td>56</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1403</th>
      <td>never-grow-old</td>
      <td>90</td>
      <td>64</td>
    </tr>
    <tr>
      <th>935</th>
      <td>charlie-says</td>
      <td>58</td>
      <td>47</td>
    </tr>
    <tr>
      <th>76</th>
      <td>big-mommas-like-father-like-son</td>
      <td>4</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>solis</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>432</th>
      <td>sherlock-gnomes</td>
      <td>26</td>
      <td>46</td>
    </tr>
    <tr>
      <th>158</th>
      <td>forger-the</td>
      <td>8</td>
      <td>44</td>
    </tr>
    <tr>
      <th>240</th>
      <td>man-down</td>
      <td>14</td>
      <td>45</td>
    </tr>
    <tr>
      <th>130</th>
      <td>gods-not-dead-2</td>
      <td>7</td>
      <td>52</td>
    </tr>
    <tr>
      <th>911</th>
      <td>lbj</td>
      <td>56</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1053</th>
      <td>10x10</td>
      <td>66</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>nancy-drew-and-the-hidden-staircase</td>
      <td>66</td>
      <td>47</td>
    </tr>
    <tr>
      <th>400</th>
      <td>woodshock</td>
      <td>24</td>
      <td>52</td>
    </tr>
    <tr>
      <th>668</th>
      <td>vincent-n-roxxy</td>
      <td>41</td>
      <td>39</td>
    </tr>
    <tr>
      <th>372</th>
      <td>men-in-black-international</td>
      <td>22</td>
      <td>41</td>
    </tr>
    <tr>
      <th>986</th>
      <td>velvet-buzzsaw</td>
      <td>61</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1395</th>
      <td>capernaum</td>
      <td>90</td>
      <td>35</td>
    </tr>
    <tr>
      <th>27</th>
      <td>poison-rose-the</td>
      <td>0</td>
      <td>47</td>
    </tr>
    <tr>
      <th>444</th>
      <td>my-big-fat-greek-wedding-2</td>
      <td>27</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>dragon-ball-super-broly</td>
      <td>83</td>
      <td>59</td>
    </tr>
    <tr>
      <th>251</th>
      <td>geostorm</td>
      <td>15</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1339</th>
      <td>i-am-richard-pryor</td>
      <td>85</td>
      <td>69</td>
    </tr>
    <tr>
      <th>792</th>
      <td>down-a-dark-hall</td>
      <td>49</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>unicorn-store</td>
      <td>63</td>
      <td>55</td>
    </tr>
    <tr>
      <th>178</th>
      <td>backtrace</td>
      <td>10</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1198</th>
      <td>nightmare-cinema</td>
      <td>76</td>
      <td>59</td>
    </tr>
    <tr>
      <th>766</th>
      <td>my-little-pony-the-movie</td>
      <td>47</td>
      <td>59</td>
    </tr>
    <tr>
      <th>391</th>
      <td>corporate-animals</td>
      <td>24</td>
      <td>38</td>
    </tr>
    <tr>
      <th>782</th>
      <td>adult-beginners</td>
      <td>48</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1262</th>
      <td>overlord</td>
      <td>80</td>
      <td>45</td>
    </tr>
    <tr>
      <th>927</th>
      <td>bullet-head</td>
      <td>58</td>
      <td>50</td>
    </tr>
    <tr>
      <th>987</th>
      <td>sicario-day-of-the-soldado</td>
      <td>62</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1509</th>
      <td>faces-places</td>
      <td>98</td>
      <td>49</td>
    </tr>
    <tr>
      <th>607</th>
      <td>dirt-the</td>
      <td>37</td>
      <td>47</td>
    </tr>
    <tr>
      <th>66</th>
      <td>reach-me</td>
      <td>3</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>snapshots</td>
      <td>71</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1278</th>
      <td>level-16</td>
      <td>81</td>
      <td>56</td>
    </tr>
    <tr>
      <th>580</th>
      <td>hollow-point-the</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>368</th>
      <td>all-the-devils-men</td>
      <td>22</td>
      <td>38</td>
    </tr>
    <tr>
      <th>57</th>
      <td>hillarys-america-the-secret-history-of-the-dem...</td>
      <td>3</td>
      <td>66</td>
    </tr>
    <tr>
      <th>1518</th>
      <td>tower</td>
      <td>98</td>
      <td>79</td>
    </tr>
    <tr>
      <th>781</th>
      <td>intruders</td>
      <td>48</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1349</th>
      <td>thoroughbreds</td>
      <td>86</td>
      <td>61</td>
    </tr>
    <tr>
      <th>1343</th>
      <td>upgrade</td>
      <td>86</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>first-man</td>
      <td>86</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1399</th>
      <td>gloria-bell</td>
      <td>90</td>
      <td>48</td>
    </tr>
    <tr>
      <th>0</th>
      <td>gallows-act-ii-the</td>
      <td>0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>361</th>
      <td>ithaca</td>
      <td>21</td>
      <td>42</td>
    </tr>
    <tr>
      <th>605</th>
      <td>inside-game</td>
      <td>37</td>
      <td>52</td>
    </tr>
    <tr>
      <th>424</th>
      <td>benefactor-the</td>
      <td>26</td>
      <td>47</td>
    </tr>
    <tr>
      <th>387</th>
      <td>overboard</td>
      <td>23</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>speed-of-life</td>
      <td>77</td>
      <td>81</td>
    </tr>
    <tr>
      <th>60</th>
      <td>shanghai</td>
      <td>3</td>
      <td>52</td>
    </tr>
    <tr>
      <th>18</th>
      <td>london-fields</td>
      <td>0</td>
      <td>67</td>
    </tr>
    <tr>
      <th>62</th>
      <td>vice</td>
      <td>3</td>
      <td>34</td>
    </tr>
    <tr>
      <th>838</th>
      <td>papillon</td>
      <td>51</td>
      <td>49</td>
    </tr>
    <tr>
      <th>127</th>
      <td>survivor</td>
      <td>7</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1083</th>
      <td>disappearance-at-clifton-hill</td>
      <td>68</td>
      <td>54</td>
    </tr>
    <tr>
      <th>59</th>
      <td>6-souls</td>
      <td>3</td>
      <td>31</td>
    </tr>
    <tr>
      <th>25</th>
      <td>maximum-impact</td>
      <td>0</td>
      <td>39</td>
    </tr>
    <tr>
      <th>147</th>
      <td>cherry-tree</td>
      <td>8</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1301</th>
      <td>creed-ii</td>
      <td>83</td>
      <td>49</td>
    </tr>
    <tr>
      <th>762</th>
      <td>kung-fu-yoga</td>
      <td>47</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>knives-and-skin</td>
      <td>66</td>
      <td>54</td>
    </tr>
    <tr>
      <th>408</th>
      <td>running-with-the-devil</td>
      <td>24</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1157</th>
      <td>christopher-robin</td>
      <td>72</td>
      <td>54</td>
    </tr>
    <tr>
      <th>671</th>
      <td>katie-says-goodbye</td>
      <td>41</td>
      <td>62</td>
    </tr>
    <tr>
      <th>339</th>
      <td>phil</td>
      <td>20</td>
      <td>64</td>
    </tr>
    <tr>
      <th>162</th>
      <td>krystal</td>
      <td>9</td>
      <td>41</td>
    </tr>
    <tr>
      <th>202</th>
      <td>action-point</td>
      <td>12</td>
      <td>41</td>
    </tr>
    <tr>
      <th>560</th>
      <td>voice-from-the-stone</td>
      <td>35</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>feast-of-the-seven-fishes</td>
      <td>87</td>
      <td>47</td>
    </tr>
    <tr>
      <th>319</th>
      <td>house-the</td>
      <td>19</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1325</th>
      <td>untouchable</td>
      <td>84</td>
      <td>76</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>diane</td>
      <td>92</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>bouncer-the</td>
      <td>79</td>
      <td>36</td>
    </tr>
    <tr>
      <th>563</th>
      <td>secret-scripture-the</td>
      <td>35</td>
      <td>51</td>
    </tr>
    <tr>
      <th>981</th>
      <td>i-can-only-imagine</td>
      <td>61</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>hustlers</td>
      <td>86</td>
      <td>37</td>
    </tr>
    <tr>
      <th>508</th>
      <td>my-all-american</td>
      <td>31</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>adopt-a-highway</td>
      <td>67</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>mission-impossible-fallout</td>
      <td>97</td>
      <td>41</td>
    </tr>
    <tr>
      <th>404</th>
      <td>host-the</td>
      <td>24</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1165</th>
      <td>claras-ghost</td>
      <td>73</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1127</th>
      <td>all-is-true</td>
      <td>71</td>
      <td>44</td>
    </tr>
    <tr>
      <th>435</th>
      <td>mortal-engines</td>
      <td>26</td>
      <td>38</td>
    </tr>
    <tr>
      <th>406</th>
      <td>open-water-3-cage-dive</td>
      <td>24</td>
      <td>41</td>
    </tr>
    <tr>
      <th>266</th>
      <td>transporter-refueled-the</td>
      <td>15</td>
      <td>51</td>
    </tr>
    <tr>
      <th>764</th>
      <td>killing-gunther</td>
      <td>47</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1134</th>
      <td>piercing</td>
      <td>71</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>lullaby-the</td>
      <td>63</td>
      <td>44</td>
    </tr>
    <tr>
      <th>166</th>
      <td>holmes-and-watson</td>
      <td>9</td>
      <td>46</td>
    </tr>
    <tr>
      <th>528</th>
      <td>an-actor-prepares</td>
      <td>32</td>
      <td>64</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>us-and-them</td>
      <td>62</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1511</th>
      <td>dont-think-twice</td>
      <td>98</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1382</th>
      <td>blinded-by-the-light</td>
      <td>88</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1285</th>
      <td>funny-story</td>
      <td>82</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1398</th>
      <td>paddleton</td>
      <td>90</td>
      <td>37</td>
    </tr>
    <tr>
      <th>889</th>
      <td>commuter-the</td>
      <td>55</td>
      <td>32</td>
    </tr>
    <tr>
      <th>222</th>
      <td>spark-a-space-tail</td>
      <td>13</td>
      <td>37</td>
    </tr>
    <tr>
      <th>295</th>
      <td>hatton-garden-job-the</td>
      <td>17</td>
      <td>39</td>
    </tr>
    <tr>
      <th>749</th>
      <td>clown</td>
      <td>46</td>
      <td>49</td>
    </tr>
    <tr>
      <th>78</th>
      <td>hangman</td>
      <td>4</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1510</th>
      <td>lady-bird</td>
      <td>98</td>
      <td>51</td>
    </tr>
    <tr>
      <th>418</th>
      <td>nun-the</td>
      <td>25</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1122</th>
      <td>nappily-ever-after</td>
      <td>70</td>
      <td>45</td>
    </tr>
    <tr>
      <th>645</th>
      <td>den-of-thieves</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>366</th>
      <td>mile-22</td>
      <td>22</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1461</th>
      <td>incredibles-2</td>
      <td>94</td>
      <td>38</td>
    </tr>
    <tr>
      <th>486</th>
      <td>rodin</td>
      <td>29</td>
      <td>42</td>
    </tr>
    <tr>
      <th>577</th>
      <td>most-hated-woman-in-america-the</td>
      <td>35</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1424</th>
      <td>giant-little-ones</td>
      <td>92</td>
      <td>40</td>
    </tr>
    <tr>
      <th>96</th>
      <td>mothers-day</td>
      <td>5</td>
      <td>35</td>
    </tr>
    <tr>
      <th>61</th>
      <td>cold-light-of-day-the</td>
      <td>3</td>
      <td>49</td>
    </tr>
    <tr>
      <th>599</th>
      <td>kidnap</td>
      <td>37</td>
      <td>26</td>
    </tr>
    <tr>
      <th>122</th>
      <td>billionaire-boys-club</td>
      <td>6</td>
      <td>39</td>
    </tr>
    <tr>
      <th>853</th>
      <td>person-to-person</td>
      <td>52</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1365</th>
      <td>plus-one</td>
      <td>87</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>possum</td>
      <td>90</td>
      <td>64</td>
    </tr>
    <tr>
      <th>284</th>
      <td>father-figures</td>
      <td>16</td>
      <td>42</td>
    </tr>
    <tr>
      <th>883</th>
      <td>half-magic</td>
      <td>54</td>
      <td>49</td>
    </tr>
    <tr>
      <th>386</th>
      <td>dark-places</td>
      <td>23</td>
      <td>38</td>
    </tr>
    <tr>
      <th>223</th>
      <td>damascus-cover</td>
      <td>13</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1317</th>
      <td>weekend-the</td>
      <td>84</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1012</th>
      <td>shes-missing</td>
      <td>63</td>
      <td>48</td>
    </tr>
    <tr>
      <th>354</th>
      <td>zoolander-2</td>
      <td>21</td>
      <td>44</td>
    </tr>
    <tr>
      <th>561</th>
      <td>6-underground</td>
      <td>35</td>
      <td>39</td>
    </tr>
    <tr>
      <th>846</th>
      <td>genius</td>
      <td>52</td>
      <td>52</td>
    </tr>
    <tr>
      <th>559</th>
      <td>masterminds</td>
      <td>34</td>
      <td>43</td>
    </tr>
    <tr>
      <th>547</th>
      <td>horrible-bosses-2</td>
      <td>34</td>
      <td>61</td>
    </tr>
    <tr>
      <th>632</th>
      <td>run-the-race</td>
      <td>39</td>
      <td>40</td>
    </tr>
    <tr>
      <th>779</th>
      <td>whered-you-go-bernadette</td>
      <td>48</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>in-search-of-greatness</td>
      <td>89</td>
      <td>80</td>
    </tr>
    <tr>
      <th>570</th>
      <td>drifter</td>
      <td>35</td>
      <td>55</td>
    </tr>
    <tr>
      <th>220</th>
      <td>when-the-bough-breaks</td>
      <td>12</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1515</th>
      <td>cameraperson</td>
      <td>98</td>
      <td>51</td>
    </tr>
    <tr>
      <th>446</th>
      <td>fathers-and-daughters</td>
      <td>27</td>
      <td>38</td>
    </tr>
    <tr>
      <th>657</th>
      <td>stuber</td>
      <td>41</td>
      <td>35</td>
    </tr>
    <tr>
      <th>32</th>
      <td>daddy-day-camp</td>
      <td>1</td>
      <td>28</td>
    </tr>
    <tr>
      <th>340</th>
      <td>daddys-home-2</td>
      <td>20</td>
      <td>49</td>
    </tr>
    <tr>
      <th>793</th>
      <td>overcomer</td>
      <td>49</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1074</th>
      <td>jay-and-silent-bob-reboot</td>
      <td>67</td>
      <td>59</td>
    </tr>
    <tr>
      <th>495</th>
      <td>inconceivable</td>
      <td>30</td>
      <td>49</td>
    </tr>
    <tr>
      <th>807</th>
      <td>55-steps</td>
      <td>49</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1183</th>
      <td>princess-switch-the</td>
      <td>74</td>
      <td>39</td>
    </tr>
    <tr>
      <th>401</th>
      <td>dont-hang-up</td>
      <td>24</td>
      <td>33</td>
    </tr>
    <tr>
      <th>376</th>
      <td>kitchen-the</td>
      <td>22</td>
      <td>39</td>
    </tr>
    <tr>
      <th>46</th>
      <td>darkness-the</td>
      <td>2</td>
      <td>48</td>
    </tr>
    <tr>
      <th>845</th>
      <td>woman-walks-ahead</td>
      <td>52</td>
      <td>54</td>
    </tr>
    <tr>
      <th>732</th>
      <td>like-father</td>
      <td>45</td>
      <td>57</td>
    </tr>
    <tr>
      <th>921</th>
      <td>despicable-me-3</td>
      <td>57</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>mega-time-squad</td>
      <td>75</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1245</th>
      <td>cold-brook</td>
      <td>79</td>
      <td>34</td>
    </tr>
    <tr>
      <th>253</th>
      <td>aloft</td>
      <td>15</td>
      <td>40</td>
    </tr>
    <tr>
      <th>378</th>
      <td>little-boy</td>
      <td>23</td>
      <td>38</td>
    </tr>
    <tr>
      <th>475</th>
      <td>corrupted-the</td>
      <td>29</td>
      <td>35</td>
    </tr>
    <tr>
      <th>342</th>
      <td>book-of-henry-the</td>
      <td>21</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1214</th>
      <td>black-47</td>
      <td>77</td>
      <td>49</td>
    </tr>
    <tr>
      <th>385</th>
      <td>fist-fight</td>
      <td>23</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>prospect</td>
      <td>88</td>
      <td>53</td>
    </tr>
    <tr>
      <th>802</th>
      <td>battle-of-jangsari-the</td>
      <td>49</td>
      <td>62</td>
    </tr>
    <tr>
      <th>20</th>
      <td>after-party-the</td>
      <td>0</td>
      <td>33</td>
    </tr>
    <tr>
      <th>966</th>
      <td>this-beautiful-fantastic</td>
      <td>61</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1379</th>
      <td>missing-link</td>
      <td>88</td>
      <td>69</td>
    </tr>
    <tr>
      <th>937</th>
      <td>you-might-be-the-killer</td>
      <td>58</td>
      <td>31</td>
    </tr>
    <tr>
      <th>773</th>
      <td>spy-who-dumped-me-the</td>
      <td>48</td>
      <td>41</td>
    </tr>
    <tr>
      <th>529</th>
      <td>devils-gate</td>
      <td>32</td>
      <td>49</td>
    </tr>
    <tr>
      <th>789</th>
      <td>bloodline</td>
      <td>49</td>
      <td>67</td>
    </tr>
    <tr>
      <th>477</th>
      <td>pirates-of-the-caribbean-dead-men-tell-no-tales</td>
      <td>29</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>rivers-edge</td>
      <td>70</td>
      <td>30</td>
    </tr>
    <tr>
      <th>180</th>
      <td>dirty-grandpa</td>
      <td>10</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1470</th>
      <td>queen-of-hearts</td>
      <td>95</td>
      <td>42</td>
    </tr>
    <tr>
      <th>170</th>
      <td>joe-dirt-2-beautiful-loser</td>
      <td>9</td>
      <td>44</td>
    </tr>
    <tr>
      <th>349</th>
      <td>bleeding-steel</td>
      <td>21</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1310</th>
      <td>dora-and-the-lost-city-of-gold</td>
      <td>83</td>
      <td>43</td>
    </tr>
    <tr>
      <th>975</th>
      <td>murder-on-the-orient-express</td>
      <td>61</td>
      <td>51</td>
    </tr>
    <tr>
      <th>797</th>
      <td>curse-of-buckout-road-the</td>
      <td>49</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1075</th>
      <td>sister-aimee</td>
      <td>67</td>
      <td>43</td>
    </tr>
    <tr>
      <th>355</th>
      <td>daughter-of-the-wolf</td>
      <td>21</td>
      <td>53</td>
    </tr>
    <tr>
      <th>437</th>
      <td>last-sharknado-its-about-time-the</td>
      <td>26</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>coco</td>
      <td>97</td>
      <td>44</td>
    </tr>
    <tr>
      <th>616</th>
      <td>mary-shelley</td>
      <td>38</td>
      <td>63</td>
    </tr>
    <tr>
      <th>696</th>
      <td>ready-to-mingle</td>
      <td>43</td>
      <td>47</td>
    </tr>
    <tr>
      <th>458</th>
      <td>between-worlds</td>
      <td>28</td>
      <td>51</td>
    </tr>
    <tr>
      <th>191</th>
      <td>eloise</td>
      <td>10</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1316</th>
      <td>vanishing-the</td>
      <td>84</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1368</th>
      <td>frankensteins-monsters-monster-frankenstein</td>
      <td>87</td>
      <td>50</td>
    </tr>
    <tr>
      <th>843</th>
      <td>complete-unknown</td>
      <td>51</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1477</th>
      <td>first-love</td>
      <td>96</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>pain-and-glory</td>
      <td>96</td>
      <td>54</td>
    </tr>
    <tr>
      <th>790</th>
      <td>sonata-the</td>
      <td>49</td>
      <td>50</td>
    </tr>
    <tr>
      <th>79</th>
      <td>outsider-the</td>
      <td>4</td>
      <td>55</td>
    </tr>
    <tr>
      <th>711</th>
      <td>seeds</td>
      <td>44</td>
      <td>74</td>
    </tr>
    <tr>
      <th>558</th>
      <td>a-dogs-purpose</td>
      <td>34</td>
      <td>39</td>
    </tr>
    <tr>
      <th>277</th>
      <td>jeepers-creepers-3</td>
      <td>16</td>
      <td>35</td>
    </tr>
    <tr>
      <th>549</th>
      <td>live-by-night</td>
      <td>34</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1388</th>
      <td>souvenir-the</td>
      <td>89</td>
      <td>50</td>
    </tr>
    <tr>
      <th>636</th>
      <td>tremors-a-cold-day-in-hell</td>
      <td>39</td>
      <td>42</td>
    </tr>
    <tr>
      <th>507</th>
      <td>catcher-was-a-spy-the</td>
      <td>31</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>zombieland-double-tap</td>
      <td>67</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1462</th>
      <td>tell-me-who-i-am</td>
      <td>95</td>
      <td>66</td>
    </tr>
    <tr>
      <th>753</th>
      <td>how-to-talk-to-girls-at-parties</td>
      <td>46</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>lasso</td>
      <td>62</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1420</th>
      <td>rockos-modern-life-static-cling</td>
      <td>91</td>
      <td>48</td>
    </tr>
    <tr>
      <th>469</th>
      <td>outlaws-and-angels</td>
      <td>28</td>
      <td>37</td>
    </tr>
    <tr>
      <th>429</th>
      <td>rambo-last-blood</td>
      <td>26</td>
      <td>62</td>
    </tr>
    <tr>
      <th>1508</th>
      <td>apollo-11</td>
      <td>98</td>
      <td>47</td>
    </tr>
    <tr>
      <th>177</th>
      <td>berlin-i-love-you</td>
      <td>10</td>
      <td>64</td>
    </tr>
    <tr>
      <th>1226</th>
      <td>revival-the</td>
      <td>78</td>
      <td>40</td>
    </tr>
    <tr>
      <th>119</th>
      <td>addicted</td>
      <td>6</td>
      <td>46</td>
    </tr>
    <tr>
      <th>438</th>
      <td>drone</td>
      <td>26</td>
      <td>30</td>
    </tr>
    <tr>
      <th>780</th>
      <td>almost-christmas</td>
      <td>48</td>
      <td>28</td>
    </tr>
    <tr>
      <th>621</th>
      <td>angel-has-fallen</td>
      <td>38</td>
      <td>41</td>
    </tr>
    <tr>
      <th>587</th>
      <td>tone-deaf</td>
      <td>36</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1311</th>
      <td>girl</td>
      <td>84</td>
      <td>47</td>
    </tr>
    <tr>
      <th>95</th>
      <td>courier-the</td>
      <td>5</td>
      <td>26</td>
    </tr>
    <tr>
      <th>172</th>
      <td>tulip-fever</td>
      <td>9</td>
      <td>46</td>
    </tr>
    <tr>
      <th>436</th>
      <td>colonia</td>
      <td>26</td>
      <td>42</td>
    </tr>
    <tr>
      <th>834</th>
      <td>freak-show</td>
      <td>51</td>
      <td>50</td>
    </tr>
    <tr>
      <th>321</th>
      <td>shes-just-a-shadow</td>
      <td>19</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1297</th>
      <td>lady-j</td>
      <td>82</td>
      <td>59</td>
    </tr>
    <tr>
      <th>658</th>
      <td>47-meters-down-uncaged</td>
      <td>41</td>
      <td>45</td>
    </tr>
    <tr>
      <th>523</th>
      <td>better-start-running</td>
      <td>32</td>
      <td>55</td>
    </tr>
    <tr>
      <th>821</th>
      <td>miss-julie</td>
      <td>51</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1425</th>
      <td>wildlife</td>
      <td>92</td>
      <td>45</td>
    </tr>
    <tr>
      <th>922</th>
      <td>alone-in-berlin</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>772</th>
      <td>a-little-chaos</td>
      <td>48</td>
      <td>44</td>
    </tr>
    <tr>
      <th>322</th>
      <td>hospitality</td>
      <td>19</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>styx</td>
      <td>94</td>
      <td>36</td>
    </tr>
    <tr>
      <th>187</th>
      <td>fifty-shades-darker</td>
      <td>10</td>
      <td>40</td>
    </tr>
    <tr>
      <th>410</th>
      <td>sleepless</td>
      <td>24</td>
      <td>25</td>
    </tr>
    <tr>
      <th>154</th>
      <td>fantastic-four</td>
      <td>8</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1173</th>
      <td>anchor-and-hope</td>
      <td>74</td>
      <td>53</td>
    </tr>
    <tr>
      <th>124</th>
      <td>reprisal</td>
      <td>7</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1315</th>
      <td>dumplin</td>
      <td>84</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1473</th>
      <td>harpoon</td>
      <td>95</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1341</th>
      <td>soni</td>
      <td>86</td>
      <td>46</td>
    </tr>
    <tr>
      <th>649</th>
      <td>a-wrinkle-in-time</td>
      <td>41</td>
      <td>57</td>
    </tr>
    <tr>
      <th>1334</th>
      <td>miseducation-of-cameron-post-the</td>
      <td>85</td>
      <td>55</td>
    </tr>
    <tr>
      <th>235</th>
      <td>autumn-lights</td>
      <td>14</td>
      <td>62</td>
    </tr>
    <tr>
      <th>729</th>
      <td>cold-skin</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>364</th>
      <td>inferno</td>
      <td>22</td>
      <td>44</td>
    </tr>
    <tr>
      <th>494</th>
      <td>mapplethorpe</td>
      <td>30</td>
      <td>58</td>
    </tr>
    <tr>
      <th>97</th>
      <td>battle-of-the-year</td>
      <td>5</td>
      <td>52</td>
    </tr>
    <tr>
      <th>718</th>
      <td>bokeh</td>
      <td>45</td>
      <td>58</td>
    </tr>
    <tr>
      <th>303</th>
      <td>incarnate</td>
      <td>18</td>
      <td>49</td>
    </tr>
    <tr>
      <th>382</th>
      <td>90-minutes-in-heaven</td>
      <td>23</td>
      <td>46</td>
    </tr>
    <tr>
      <th>689</th>
      <td>christmas-inheritance</td>
      <td>43</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>a-private-war</td>
      <td>87</td>
      <td>40</td>
    </tr>
    <tr>
      <th>586</th>
      <td>jack-reacher-never-go-back</td>
      <td>36</td>
      <td>28</td>
    </tr>
    <tr>
      <th>17</th>
      <td>enter-the-anime</td>
      <td>0</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1039</th>
      <td>danger-close</td>
      <td>65</td>
      <td>69</td>
    </tr>
    <tr>
      <th>851</th>
      <td>3-from-hell</td>
      <td>52</td>
      <td>25</td>
    </tr>
    <tr>
      <th>806</th>
      <td>age-of-summer</td>
      <td>49</td>
      <td>51</td>
    </tr>
    <tr>
      <th>53</th>
      <td>robocop-3</td>
      <td>2</td>
      <td>36</td>
    </tr>
    <tr>
      <th>687</th>
      <td>driverx</td>
      <td>43</td>
      <td>63</td>
    </tr>
    <tr>
      <th>190</th>
      <td>gods-not-dead-a-light-in-darkness</td>
      <td>10</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1485</th>
      <td>come-as-you-are</td>
      <td>96</td>
      <td>51</td>
    </tr>
    <tr>
      <th>970</th>
      <td>arctic-tale</td>
      <td>61</td>
      <td>59</td>
    </tr>
    <tr>
      <th>705</th>
      <td>yellow-birds-the</td>
      <td>44</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>quiet-one-the</td>
      <td>66</td>
      <td>80</td>
    </tr>
    <tr>
      <th>535</th>
      <td>fun-mom-dinner</td>
      <td>33</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1224</th>
      <td>captain-marvel</td>
      <td>77</td>
      <td>44</td>
    </tr>
    <tr>
      <th>964</th>
      <td>time-trap</td>
      <td>60</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1405</th>
      <td>a-star-is-born</td>
      <td>90</td>
      <td>53</td>
    </tr>
    <tr>
      <th>553</th>
      <td>vhs-viral</td>
      <td>34</td>
      <td>23</td>
    </tr>
    <tr>
      <th>71</th>
      <td>because-i-said-so</td>
      <td>3</td>
      <td>40</td>
    </tr>
    <tr>
      <th>109</th>
      <td>amityville-murders-the</td>
      <td>6</td>
      <td>51</td>
    </tr>
    <tr>
      <th>765</th>
      <td>jurassic-world-fallen-kingdom</td>
      <td>47</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1188</th>
      <td>pledge</td>
      <td>75</td>
      <td>49</td>
    </tr>
    <tr>
      <th>738</th>
      <td>rock-dog</td>
      <td>46</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>swinging-safari</td>
      <td>76</td>
      <td>40</td>
    </tr>
    <tr>
      <th>88</th>
      <td>paul-blart-mall-cop-2</td>
      <td>4</td>
      <td>34</td>
    </tr>
    <tr>
      <th>629</th>
      <td>justice-league</td>
      <td>39</td>
      <td>44</td>
    </tr>
    <tr>
      <th>1220</th>
      <td>devils-doorway-the</td>
      <td>77</td>
      <td>47</td>
    </tr>
    <tr>
      <th>810</th>
      <td>ladyworld</td>
      <td>49</td>
      <td>50</td>
    </tr>
    <tr>
      <th>313</th>
      <td>wish-upon</td>
      <td>18</td>
      <td>38</td>
    </tr>
    <tr>
      <th>708</th>
      <td>girls-of-the-sun</td>
      <td>44</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1503</th>
      <td>parasite</td>
      <td>98</td>
      <td>50</td>
    </tr>
    <tr>
      <th>163</th>
      <td>stonewall</td>
      <td>9</td>
      <td>37</td>
    </tr>
    <tr>
      <th>212</th>
      <td>no-good-deed</td>
      <td>12</td>
      <td>34</td>
    </tr>
    <tr>
      <th>556</th>
      <td>24-exposures</td>
      <td>34</td>
      <td>61</td>
    </tr>
    <tr>
      <th>123</th>
      <td>grown-ups-2</td>
      <td>6</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1502</th>
      <td>moonlight</td>
      <td>98</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1312</th>
      <td>sollers-point</td>
      <td>84</td>
      <td>61</td>
    </tr>
    <tr>
      <th>829</th>
      <td>12-strong</td>
      <td>51</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1342</th>
      <td>sisters-brothers-the</td>
      <td>86</td>
      <td>37</td>
    </tr>
    <tr>
      <th>837</th>
      <td>tomb-raider</td>
      <td>51</td>
      <td>40</td>
    </tr>
    <tr>
      <th>308</th>
      <td>underworld-blood-wars</td>
      <td>18</td>
      <td>30</td>
    </tr>
    <tr>
      <th>449</th>
      <td>aftermath-the</td>
      <td>27</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>welcome-to-mercy</td>
      <td>66</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1299</th>
      <td>art-of-self-defense-the</td>
      <td>83</td>
      <td>49</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>mustang-the</td>
      <td>93</td>
      <td>49</td>
    </tr>
    <tr>
      <th>201</th>
      <td>first-kill</td>
      <td>11</td>
      <td>40</td>
    </tr>
    <tr>
      <th>582</th>
      <td>glass</td>
      <td>36</td>
      <td>55</td>
    </tr>
    <tr>
      <th>633</th>
      <td>super-the</td>
      <td>39</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1213</th>
      <td>last-flag-flying</td>
      <td>77</td>
      <td>54</td>
    </tr>
    <tr>
      <th>695</th>
      <td>wedding-guest-the</td>
      <td>43</td>
      <td>49</td>
    </tr>
    <tr>
      <th>788</th>
      <td>tolkien</td>
      <td>49</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1113</th>
      <td>king-the</td>
      <td>70</td>
      <td>69</td>
    </tr>
    <tr>
      <th>197</th>
      <td>preppie-connection-the</td>
      <td>11</td>
      <td>36</td>
    </tr>
    <tr>
      <th>233</th>
      <td>nine-lives</td>
      <td>14</td>
      <td>50</td>
    </tr>
    <tr>
      <th>183</th>
      <td>basmati-blues</td>
      <td>10</td>
      <td>62</td>
    </tr>
    <tr>
      <th>427</th>
      <td>week-of-the</td>
      <td>26</td>
      <td>34</td>
    </tr>
    <tr>
      <th>650</th>
      <td>spinning-man</td>
      <td>41</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1030</th>
      <td>resistance</td>
      <td>64</td>
      <td>57</td>
    </tr>
    <tr>
      <th>1488</th>
      <td>hale-county-this-morning-this-evening</td>
      <td>96</td>
      <td>54</td>
    </tr>
    <tr>
      <th>412</th>
      <td>perfect-match-the</td>
      <td>25</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1374</th>
      <td>framing-john-delorean</td>
      <td>88</td>
      <td>75</td>
    </tr>
    <tr>
      <th>979</th>
      <td>burn-your-maps</td>
      <td>61</td>
      <td>55</td>
    </tr>
    <tr>
      <th>562</th>
      <td>fantastic-beasts-the-crimes-of-grindelwald</td>
      <td>35</td>
      <td>33</td>
    </tr>
    <tr>
      <th>108</th>
      <td>don-peyote</td>
      <td>6</td>
      <td>44</td>
    </tr>
    <tr>
      <th>447</th>
      <td>rebel-in-the-rye</td>
      <td>27</td>
      <td>60</td>
    </tr>
    <tr>
      <th>714</th>
      <td>rough-night</td>
      <td>44</td>
      <td>41</td>
    </tr>
    <tr>
      <th>825</th>
      <td>boulevard</td>
      <td>51</td>
      <td>44</td>
    </tr>
    <tr>
      <th>360</th>
      <td>jem-and-the-holograms</td>
      <td>21</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>brink-the</td>
      <td>83</td>
      <td>63</td>
    </tr>
    <tr>
      <th>942</th>
      <td>bad-moms</td>
      <td>58</td>
      <td>57</td>
    </tr>
    <tr>
      <th>860</th>
      <td>bad-samaritan</td>
      <td>52</td>
      <td>56</td>
    </tr>
    <tr>
      <th>1373</th>
      <td>my-name-is-myeisha</td>
      <td>88</td>
      <td>55</td>
    </tr>
    <tr>
      <th>439</th>
      <td>duel-the</td>
      <td>26</td>
      <td>42</td>
    </tr>
    <tr>
      <th>168</th>
      <td>all-relative</td>
      <td>9</td>
      <td>46</td>
    </tr>
    <tr>
      <th>369</th>
      <td>nerdland</td>
      <td>22</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1360</th>
      <td>linda-ronstadt-the-sound-of-my-voice</td>
      <td>87</td>
      <td>84</td>
    </tr>
    <tr>
      <th>716</th>
      <td>allure</td>
      <td>44</td>
      <td>69</td>
    </tr>
    <tr>
      <th>1038</th>
      <td>little-stranger-the</td>
      <td>65</td>
      <td>58</td>
    </tr>
    <tr>
      <th>324</th>
      <td>beneath-the-leaves</td>
      <td>19</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1168</th>
      <td>bad-times-at-the-el-royale</td>
      <td>74</td>
      <td>42</td>
    </tr>
    <tr>
      <th>466</th>
      <td>red-pill-the</td>
      <td>28</td>
      <td>73</td>
    </tr>
    <tr>
      <th>397</th>
      <td>solace</td>
      <td>24</td>
      <td>33</td>
    </tr>
    <tr>
      <th>231</th>
      <td>circle-the</td>
      <td>13</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1281</th>
      <td>someone-great</td>
      <td>81</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1243</th>
      <td>code-8</td>
      <td>79</td>
      <td>36</td>
    </tr>
    <tr>
      <th>652</th>
      <td>star-the</td>
      <td>41</td>
      <td>42</td>
    </tr>
    <tr>
      <th>938</th>
      <td>grinch-the</td>
      <td>58</td>
      <td>54</td>
    </tr>
    <tr>
      <th>976</th>
      <td>shed-the</td>
      <td>61</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>day-shall-come-the</td>
      <td>64</td>
      <td>37</td>
    </tr>
    <tr>
      <th>745</th>
      <td>31</td>
      <td>46</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>blood-on-her-name</td>
      <td>94</td>
      <td>45</td>
    </tr>
    <tr>
      <th>320</th>
      <td>mandela-effect-the</td>
      <td>19</td>
      <td>47</td>
    </tr>
    <tr>
      <th>48</th>
      <td>bless-the-child</td>
      <td>2</td>
      <td>41</td>
    </tr>
    <tr>
      <th>552</th>
      <td>welcome-to-marwen</td>
      <td>34</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1331</th>
      <td>braid</td>
      <td>85</td>
      <td>49</td>
    </tr>
    <tr>
      <th>107</th>
      <td>fifty-shades-of-black</td>
      <td>6</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1486</th>
      <td>border</td>
      <td>96</td>
      <td>53</td>
    </tr>
    <tr>
      <th>715</th>
      <td>mary-magdalene</td>
      <td>44</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1246</th>
      <td>luciferina</td>
      <td>79</td>
      <td>36</td>
    </tr>
    <tr>
      <th>640</th>
      <td>maleficent-mistress-of-evil</td>
      <td>39</td>
      <td>41</td>
    </tr>
    <tr>
      <th>721</th>
      <td>asher</td>
      <td>45</td>
      <td>62</td>
    </tr>
    <tr>
      <th>573</th>
      <td>anna</td>
      <td>35</td>
      <td>55</td>
    </tr>
    <tr>
      <th>460</th>
      <td>trick</td>
      <td>28</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>downton-abbey</td>
      <td>83</td>
      <td>49</td>
    </tr>
    <tr>
      <th>953</th>
      <td>new-money</td>
      <td>59</td>
      <td>53</td>
    </tr>
    <tr>
      <th>660</th>
      <td>song-of-sway-lake-the</td>
      <td>41</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1081</th>
      <td>bottom-of-the-9th</td>
      <td>68</td>
      <td>32</td>
    </tr>
    <tr>
      <th>957</th>
      <td>satanic-panic</td>
      <td>60</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1163</th>
      <td>death-of-dick-long-the</td>
      <td>73</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1377</th>
      <td>rocketman</td>
      <td>88</td>
      <td>36</td>
    </tr>
    <tr>
      <th>614</th>
      <td>mountain-between-us-the</td>
      <td>38</td>
      <td>53</td>
    </tr>
    <tr>
      <th>10</th>
      <td>saving-zoe</td>
      <td>0</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>portals</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>875</th>
      <td>rules-dont-apply</td>
      <td>54</td>
      <td>51</td>
    </tr>
    <tr>
      <th>735</th>
      <td>bad-batch-the</td>
      <td>46</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>wildling</td>
      <td>71</td>
      <td>50</td>
    </tr>
    <tr>
      <th>141</th>
      <td>exposed</td>
      <td>7</td>
      <td>67</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>wonders-of-the-sea</td>
      <td>75</td>
      <td>75</td>
    </tr>
    <tr>
      <th>814</th>
      <td>wounds</td>
      <td>50</td>
      <td>51</td>
    </tr>
    <tr>
      <th>820</th>
      <td>interview-the</td>
      <td>51</td>
      <td>47</td>
    </tr>
    <tr>
      <th>35</th>
      <td>left-behind</td>
      <td>1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1131</th>
      <td>hostiles</td>
      <td>71</td>
      <td>51</td>
    </tr>
    <tr>
      <th>301</th>
      <td>rupture</td>
      <td>18</td>
      <td>46</td>
    </tr>
    <tr>
      <th>434</th>
      <td>sandy-wexler</td>
      <td>26</td>
      <td>66</td>
    </tr>
    <tr>
      <th>800</th>
      <td>american-son</td>
      <td>49</td>
      <td>48</td>
    </tr>
    <tr>
      <th>896</th>
      <td>beach-bum-the</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1029</th>
      <td>stellas-last-weekend</td>
      <td>64</td>
      <td>64</td>
    </tr>
    <tr>
      <th>944</th>
      <td>scarborough</td>
      <td>59</td>
      <td>48</td>
    </tr>
    <tr>
      <th>125</th>
      <td>are-you-here</td>
      <td>7</td>
      <td>54</td>
    </tr>
    <tr>
      <th>352</th>
      <td>222</td>
      <td>21</td>
      <td>41</td>
    </tr>
    <tr>
      <th>947</th>
      <td>candy-corn</td>
      <td>59</td>
      <td>33</td>
    </tr>
    <tr>
      <th>579</th>
      <td>wetlands</td>
      <td>35</td>
      <td>44</td>
    </tr>
    <tr>
      <th>454</th>
      <td>bright</td>
      <td>27</td>
      <td>33</td>
    </tr>
    <tr>
      <th>595</th>
      <td>donnybrook</td>
      <td>37</td>
      <td>48</td>
    </tr>
    <tr>
      <th>550</th>
      <td>cut-bank</td>
      <td>34</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>red-letter-day</td>
      <td>66</td>
      <td>34</td>
    </tr>
    <tr>
      <th>350</th>
      <td>6-below-miracle-on-the-mountain</td>
      <td>21</td>
      <td>31</td>
    </tr>
    <tr>
      <th>467</th>
      <td>curse-of-la-llorona-the</td>
      <td>28</td>
      <td>42</td>
    </tr>
    <tr>
      <th>868</th>
      <td>crash-pad</td>
      <td>53</td>
      <td>37</td>
    </tr>
    <tr>
      <th>928</th>
      <td>liam-gallagher-as-it-was</td>
      <td>58</td>
      <td>50</td>
    </tr>
    <tr>
      <th>278</th>
      <td>after</td>
      <td>16</td>
      <td>63</td>
    </tr>
    <tr>
      <th>216</th>
      <td>last-knights</td>
      <td>12</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1073</th>
      <td>finding-your-feet</td>
      <td>67</td>
      <td>55</td>
    </tr>
    <tr>
      <th>1275</th>
      <td>mid90s</td>
      <td>81</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1422</th>
      <td>echo-in-the-canyon</td>
      <td>91</td>
      <td>72</td>
    </tr>
    <tr>
      <th>1376</th>
      <td>arctic</td>
      <td>88</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1351</th>
      <td>tully</td>
      <td>86</td>
      <td>52</td>
    </tr>
    <tr>
      <th>722</th>
      <td>discovery-the</td>
      <td>45</td>
      <td>49</td>
    </tr>
    <tr>
      <th>930</th>
      <td>money-monster</td>
      <td>58</td>
      <td>39</td>
    </tr>
    <tr>
      <th>894</th>
      <td>itsy-bitsy</td>
      <td>55</td>
      <td>43</td>
    </tr>
    <tr>
      <th>663</th>
      <td>girls-with-balls</td>
      <td>41</td>
      <td>53</td>
    </tr>
    <tr>
      <th>165</th>
      <td>kill-switch</td>
      <td>9</td>
      <td>31</td>
    </tr>
    <tr>
      <th>609</th>
      <td>cruise</td>
      <td>37</td>
      <td>49</td>
    </tr>
    <tr>
      <th>651</th>
      <td>phoenix-forgotten</td>
      <td>41</td>
      <td>34</td>
    </tr>
    <tr>
      <th>923</th>
      <td>what-happened-to-monday</td>
      <td>57</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1407</th>
      <td>widows</td>
      <td>90</td>
      <td>38</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>use-me</td>
      <td>85</td>
      <td>57</td>
    </tr>
    <tr>
      <th>157</th>
      <td>despite-the-falling-snow</td>
      <td>8</td>
      <td>35</td>
    </tr>
    <tr>
      <th>855</th>
      <td>blood-fest</td>
      <td>52</td>
      <td>51</td>
    </tr>
    <tr>
      <th>933</th>
      <td>i-am-the-pretty-thing-that-lives-in-the-house</td>
      <td>58</td>
      <td>79</td>
    </tr>
    <tr>
      <th>373</th>
      <td>yoga-hosers</td>
      <td>22</td>
      <td>44</td>
    </tr>
    <tr>
      <th>906</th>
      <td>154</td>
      <td>56</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>tribe-the</td>
      <td>70</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1289</th>
      <td>my-dinner-with-herve</td>
      <td>82</td>
      <td>55</td>
    </tr>
    <tr>
      <th>276</th>
      <td>blood-brother</td>
      <td>16</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>alpha</td>
      <td>79</td>
      <td>88</td>
    </tr>
    <tr>
      <th>990</th>
      <td>paradise-hills</td>
      <td>62</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1202</th>
      <td>clovehitch-killer-the</td>
      <td>76</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1266</th>
      <td>judy</td>
      <td>81</td>
      <td>54</td>
    </tr>
    <tr>
      <th>866</th>
      <td>strange-ones-the</td>
      <td>53</td>
      <td>66</td>
    </tr>
    <tr>
      <th>74</th>
      <td>legend-of-hercules-the</td>
      <td>3</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>damsel</td>
      <td>68</td>
      <td>50</td>
    </tr>
    <tr>
      <th>299</th>
      <td>ice-age-collision-course</td>
      <td>18</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>bikram-yogi-guru-predator</td>
      <td>95</td>
      <td>79</td>
    </tr>
    <tr>
      <th>37</th>
      <td>getaway</td>
      <td>1</td>
      <td>48</td>
    </tr>
    <tr>
      <th>348</th>
      <td>big-kill</td>
      <td>21</td>
      <td>39</td>
    </tr>
    <tr>
      <th>500</th>
      <td>kin</td>
      <td>31</td>
      <td>52</td>
    </tr>
    <tr>
      <th>451</th>
      <td>batman-v-superman-dawn-of-justice</td>
      <td>27</td>
      <td>42</td>
    </tr>
    <tr>
      <th>306</th>
      <td>possession-of-hannah-grace-the</td>
      <td>18</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1020</th>
      <td>parts-you-lose-the</td>
      <td>64</td>
      <td>41</td>
    </tr>
    <tr>
      <th>644</th>
      <td>where-hands-touch</td>
      <td>40</td>
      <td>48</td>
    </tr>
    <tr>
      <th>824</th>
      <td>accountant-the</td>
      <td>51</td>
      <td>39</td>
    </tr>
    <tr>
      <th>115</th>
      <td>paranoia</td>
      <td>6</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1194</th>
      <td>in-the-fade</td>
      <td>76</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1353</th>
      <td>golem-the</td>
      <td>87</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1025</th>
      <td>i-am-michael</td>
      <td>64</td>
      <td>52</td>
    </tr>
    <tr>
      <th>520</th>
      <td>jigsaw</td>
      <td>32</td>
      <td>33</td>
    </tr>
    <tr>
      <th>733</th>
      <td>dinner-the</td>
      <td>45</td>
      <td>34</td>
    </tr>
    <tr>
      <th>742</th>
      <td>dumbo</td>
      <td>46</td>
      <td>50</td>
    </tr>
    <tr>
      <th>1408</th>
      <td>how-to-train-your-dragon-the-hidden-world</td>
      <td>90</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1354</th>
      <td>sgt-stubby-an-american-hero</td>
      <td>87</td>
      <td>64</td>
    </tr>
    <tr>
      <th>462</th>
      <td>passage-to-mars</td>
      <td>28</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1256</th>
      <td>auggie</td>
      <td>79</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1480</th>
      <td>gangster-the-cop-the-devil-the</td>
      <td>96</td>
      <td>46</td>
    </tr>
    <tr>
      <th>279</th>
      <td>swiped</td>
      <td>16</td>
      <td>41</td>
    </tr>
    <tr>
      <th>300</th>
      <td>assassins-creed</td>
      <td>18</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>klaus</td>
      <td>93</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = sm.graphics.qqplot(test_residuals, dist=stats.norm, line='45', fit=True)
fig.show()
```

    //anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: UserWarning:
    
    Matplotlib is currently using module://ipykernel.pylab.backend_inline, which is a non-GUI backend, so cannot show the figure.
    



![png](main_files/main_241_1.png)


### TFIDF with Features with XG Boost Regressor


```python
X = predictors_all
```


```python
tfidf = TfidfVectorizer(max_features=5000, max_df=.9, min_df=.1, 
                        ngram_range=(1,2))
X2 = tfidf.fit_transform(rotten_model.no_stop)
```


```python
# Creating a sparse DataFrame to house both the features and the 
# processed text.
X_temp = pd.SparseDataFrame(X2, columns=tfidf.get_feature_names(),
                           default_fill_value=0)
# Necessary for next step.
X = X.reset_index(drop=True)

# Combining text matrix with script attributes.
for column in X:
    X_temp[column] = X[column]
X = X_temp
```


```python
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X), y, random_state=42,
                                                    test_size=.3)
```


```python
scaler = StandardScaler()
scaled_X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                          columns = X_train.columns)
scaled_X_test = pd.DataFrame(scaler.transform(X_test), 
                            columns=X_test.columns)
```


```python
scaled_y_train = scaler.fit_transform(np.array(y_train).reshape(-1,1))
scaled_y_test = scaler.transform(np.array(y_test).reshape(-1,1))
scaled_y_train = pd.Series(scaled_y_train.reshape(-1,), name='rotten_scores')
scaled_y_test = pd.Series(scaled_y_test.reshape(-1,), name='rotten_scores')
```


```python
scaled_X_train.fillna(0, inplace=True)
scaled_X_test.fillna(0, inplace=True)
```


```python
def test(x):
    if x == 0:
        return math.log((X + 1) / 100)
    else:
        return math.log(x / 100)
```


```python
model = XGBRegressor(random_state=42, n_estimators=100, 
                     objective='reg:squarederror')
model.fit(scaled_X_train, y_train)
```

    //anaconda3/lib/python3.7/site-packages/xgboost/core.py:587: FutureWarning:
    
    Series.base is deprecated and will be removed in a future version
    





    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0,
                 importance_type='gain', learning_rate=0.1, max_delta_step=0,
                 max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:squarederror',
                 random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 seed=None, silent=None, subsample=1, verbosity=1)




```python
y_hat_train = model.predict(pd.DataFrame(scaled_X_train))
y_hat_test = model.predict(pd.DataFrame(scaled_X_test))
```


```python
train_residuals = y_hat_train - list(y_train)
test_residuals = y_hat_test - list(y_test)
```


```python
mse_train = np.sum((y_train-y_hat_train)**2)/len(y_train)
mse_test = np.sum((y_test-y_hat_test)**2)/len(y_test)
print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)
```

    Train Mean Squarred Error: 0.054401228515797505
    Test Mean Squarred Error: 0.23082222104931627



```python
print('r^2 score:', r2_score(y_test,y_hat_test))
```

    r^2 score: 0.07667167249770279


Overall, the regression didn't work out that well. I probably could have done more with it with more time, and it would be interesting to get deeper into it. 

# Interpretation

Using the above attributes and more, I was able to create a predictive models that would ascertain whether a script was from a top-rated movie or a bottom-rated movie about 65% of the time. Within that models was some flexibility, however. Some models were better at predicting whether a movie would be 'good' (a true positive), while others were better at predicting whether a movie would be 'bad' (a true negative). The purpose of the prediction would dictate which model to employ.

Some recommendations on what can be done with this information:

Studios looking to cull out poor scripts from their stock should use the TFIDF with neural network model, which has an 81% rating for correctly predicting a bad scripts, and a 64% accuracy rating overall.

Filmmakers looking create the largest set of good scripts from the total should use the random boost with script attributes model, which has a 69% chance of correctly predicting a good script, though only a 55% accuracy score overall.

Filmmakers and screenwriters looking to find the most balanced filter for finding good screenplays and culling out the bad should look to the stacked model with support vector classifier using TFIDF followed by a random forest classifier using script attributes, which had the highest overall accuracy rate of 68%. If you're a screenwriter seeing your work coming up on the bad side, and especially if it contains a number of the "thumbs down" words noted above, you might want to start rethinking your life choices.


The scope of this idea is large, and there are numerous opportunities for further study. A few I would like to puruse are:

Using TFIDF on the word clouds to get a different, possibly more accurate take on what words are realistically more prevalent throughout a large body of texts.

Continue modeling with neural networks, especially with GloVe or other existing word embedding libraries.

Continue further work on regression analysis to predict no just whether a scripts is good or bad, but to predict the actual score it will receive on a scale of 0 to 100.

In the dashboard, provide a text entry or upload option where a script could be entered and a 'good' or 'bad' rating could be assigned, to easily determine where a given script might lie.

# Appendix: Functions

```
def format_titles(title_list):
    """This function formats the movie titles in such a way that they can be
    discerned by the web site where the screenplays will be taken from.
    
    Parameter:
    
        title_list: list
        list of titles to be formatted, generally from metacritic.
        
    Returns: 
    
        list containing titles in the proper format for scraping screenplays from
        springfieldspringfield.co.uk."""
    
    # Initializing list for later
    titles_formatted = []
    
    # Will cycle through all titles and leave them in the correct format for
    # later use.
    for title in title_list:
        title = title.lower()
        
        # Titles on this site have ', the' at the end.
        if title[:3] == 'the':
            title = title[4:] + ', the'
        
        # Getting rid of punctuation that wouldn't be in the url.
        punctuations = """!()-[]{};:'"\,<>./?@#$%^&*~"""
        for x in title: 
            if x in punctuations: 
                title = title.replace(x, '')
        
        # In the url, the spaces are hyphens.
        for x in title:
            title = title.replace('  ', '-')
            title = title.replace(' ', '-')
            title = title.replace('_', '-')
            
        titles_formatted.append(title)

    return titles_formatted

def grab_screenplays(formatted_titles):
    """Function takes correctly formatted titles and scrapes the associated
    screenplays from http://www.springfieldspringfield.co.uk.
    
    Parameter:
    
        formatted_titles: list
            Takes titles formatted by the fomatted_titles function and 
            retrieves screenplays from springfieldspringfield.
 
     Returns:
         
         list containing screenplays from argument title list.
         
         list conaining list of movies for which screenplays could not be
         found."""
         
    # scrs will hold the screenplays themselves. errors will house the titles
    # of the movies we couldn't retrieve scripts for.
    scrs = {}
    errors = []
    
    for title in formatted_titles:
        print(title)
        
        # Beginning the web scraping by opening the connection to the site,
        # putting the title into the url, and and copying the html into local variables.
        # Note the titles are pre-formatted as word-word-word-the (if there is a 'the.'
        html_page = requests.get(
            'https://www.springfieldspringfield.co.uk/movie_script.php?movie={}'.format(title))
        soup = BeautifulSoup(html_page.content, 'html.parser')
        
        # Getting the container for the screenplay.
        scr = soup.find('div', class_ = 'scrolling-script-container')
        
        # Several conditionals here to cycle through the various possibilities
        # of title formatting to find if there is a page at all.
        if scr:
            scr = scr.contents
        if not scr:
            
            # The formatting function puts 'the' at the end if there is one. 
            # Sometimes 'the' is simply removed. This will catch those.
            no_the = title[:-4]
            html_page = requests.get(
                'https://www.springfieldspringfield.co.uk/movie_script.php?movie={}'.format(no_the))
            soup = BeautifulSoup(html_page.content, 'html.parser')
            scr = soup.find('div', class_ = 'scrolling-script-container')
            if scr:
                scr = scr.contents
                
            
        if not scr:
            
            # A few movies keep 'the' at the beginning. This will catch those.
            pre_the = 'the-' + title[:-4]
            html_page = requests.get(
                'https://www.springfieldspringfield.co.uk/movie_script.php?movie={}'.format(pre_the))
            soup = BeautifulSoup(html_page.content, 'html.parser')
            scr = soup.find('div', class_ = 'scrolling-script-container')
            if scr:
                scr = scr.contents
                
            # If we still can't find the script, return a message that we couldn't
            # and put the title in an error list.
        if not scr:
            print('***The following screenplay could not be retrieved: ', title)
            errors.append(title)
        if scr:
            
            # Pulling out the unnecessary line html line breaks.
            scr = [i for i in scr if str(i) not in ['<br/>']] 
        scrs[title] = scr
        
        # Sleeping between each loop so the site server doesn't think we're 
        # trying to do anything malicious.
        time.sleep(1)
        
    return scrs, errors
    
def clean(word_list):
    """Combines each script into one long string and removes punctuation and
    html tags from the scripts. Note that this function is not always used,
    depending on what is needed from the data.
    
    Parameters:
    
        word_list: list
            List of words from one screenplay, usually in raw format
            
    Returns:
    
        A complete screenplay made of one long string with symbols and html
        breaks removed."""
    
    result = ''
    
    # Each line of the scipt is broken into a separate string. This combines
    # them into one long string.
    for line in word_list:
        result += str(line)
        
    # This removes the line breaks. They should have already been removed 
    # above but this is a backup.
    while '<br/>' in result:
        result = result.replace('<br/>', '')
        
    # Removing punctuations.
    punctuations = """!()-[]{};:"\,<>./?@#$%^&*_~"""
    for char in result: 
        if char in punctuations: 
            result = result.replace(char, '')
    
    return result

def script_classifiers(X, y, classifier, test_size=.3, cmap=plt.cm.Reds, use_tfidf=True,
                       use_split_resample=True):
    """This function takes in values for a classifier and runs them through
    a pipe. Generates scores and a confusion matrix plot.
    
    Parameters:
        
        X: DataFrame
            Features and values to be used in model.
            
        y: Series
            Target variable for classifying.
            
        Classifier: Classifier
            Which classifier to be used for training the model.
        
        cmap: pyplot color map
            Which color map to use for the confusion matrix plot.
            
        use_tfidf: bool
            Whether to use tfidf on feature data prior to running model."""
    
    # Typical train test split to retain data for validation.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=test_size,
                                                    random_state=42)
    
    # There a few more bad scripts than good ones, so I'll make them even.
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(pd.DataFrame(X_train), y_train)

    # This is necessary for putting X into the correct format if there is only 
    # one column, as it comes out of the resampler a DataFrame when the
    # classifier actuallyl wants a series if its only one column.
    if len(X_resampled.columns) < 2:
        X_resampled = X_resampled.iloc[:, 0]
    
    # Provides the option of using tfidf in case this is just using scripts 
    # attributes and not text embedding or vectorization
    if use_tfidf == False:
        pipe = Pipeline([('clf', classifier)])
    else:
        
        # These tfidf hyper-parameters have been tested via grid-search
        # elsewhere and found to be optimal.
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_df=.95, min_df=.1,
                                                  max_features=5000,
                                                  ngram_range=(1,2))), 
                         ('clf', classifier)])
        
#     scaler = StandardScaler
#     X_resampled_scaled = scaler.fit_transform(X_resampled)
#     X_test_scaled = scaler.transform(X_test)

    # Fitting the pipeline containing the tfidf processor and classifier.
    pipe.fit(X_resampled, y_resampled)
    
    # Creating predicted data.
    y_pred = pipe.predict(X_test)
    
    # Running metrics and creating a confusion matrix visual.
    print(classification_report(y_test, y_pred))

    confusion = confusion_matrix(y_test, y_pred)
    
    plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
                      show_normed=True, cmap=cmap);
    plt.show();
    
    # If there is a decision function in this classifier, we'll use it 
    # to create an ROC AUC graph.
    try:
        y_score = pipe.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print('roc_auc score: ', roc_auc)

        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show();
    except:
        pass
    
#     # Creating a decision tree classifier tree graph because it's cool.
# #     if classifier == DecisionTreeClassifier():
#     dot_data = export_graphviz(classifier, out_file=None, 
# #                                    feature_names=X.columns, 
#                                class_names=np.unique(y).astype('str'), 
#                                filled=True, rounded=True, 
#                                special_characters=True)

#     # Draw graph
#     graph = graph_from_dot_data(dot_data)  
#     # Show graph
#     Image(graph.create_png()) 
#     return Image(graph.create_png()) 

def stacked_classifier(X, y, classifier, cmap=plt.cm.Reds, use_tfidf=True):
    """This function takes in values for a classifier and runs them through
    a pipe. Used specifically to geenrate the first model for a stacked set.
    
    Parameters:
        
        X: DataFrame
            Features and values to be used in model.
            
        y: Series
            Target variable for classifying.
            
        Classifier: Classifier
            Which classifier to be used for training the model.
        
        cmap: pyplot color map
            Which color map to use for the confusion matrix plot.
            
        use_tfidf: bool
            Whether to use tfidf on feature data prior to running model.
            
    Returns:
        numpy array of predictions."""
    
    # Provides the option of using tfidf in case this is just using scripts 
    # attributes and not text embedding or vectorization
    if use_tfidf == False:
        pipe = Pipeline([('clf', classifier)])
    else:
        
        # These tfidf hyper-parameters have been tested via grid-search
        # elsewhere and found to be optimal.
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_df=.95, min_df=.1,
                                                  max_features=5000,
                                                  ngram_range=(1,2))), 
                         ('clf', classifier)])
    
    # Fitting the pipeline containing the tfidf processor and classifier.
    pipe.fit(X, y)
    
    # Creating predicted data.
    y_pred = pipe.predict(X)
    
    # Running metrics and creating a confusion matrix visual of training data.
    print(classification_report(y, y_pred))

    confusion = confusion_matrix(y, y_pred)
    
    plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
                      show_normed=True, cmap=cmap);
    plt.show();
    
    # If there is a decision function in this classifier, we'll use it 
    # to create an ROC AUC graph.
    try:
        y_score = pipe.decision_function(X)
        fpr, tpr, thresholds = roc_curve(y, y_score)
        roc_auc = auc(fpr, tpr)
        print('roc_auc score: ', roc_auc)

        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show();
    except:
        pass

    return y_pred

def hybrid_classifiers(X_train, X_test, y_train, y_test, classifier, 
                       cmap=plt.cm.Reds, use_tfidf=True):
    """This function takes in values for a classifier and runs them through
    a pipe. Generates scores and a confusion matrix plot.
    
    Parameters:
        
        X: DataFrame
            Features and values to be used in model.
            
        y: Series
            Target variable for classifying.
            
        Classifier: Classifier
            Which classifier to be used for training the model.
        
        cmap: pyplot color map
            Which color map to use for the confusion matrix plot.
            
        use_tfidf: bool
            Whether to use tfidf on feature data prior to running model."""
    
    # Typical train test split to retain data for validation.
#     X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=.3,
#                                                     random_state=42)
    
    # There a few more bad scripts than good ones, so I'll make them even.
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(pd.DataFrame(X_train), y_train)
    
    # This is necessary for putting X into the correct format if there is only 
    # one column, as it comes out of the resampler a DataFrame when the
    # classifier actuallyl wants a series if its only one column.
    if len(X_resampled.columns) < 2:
        X_resampled = X_resampled.iloc[:, 0]
    
    # Provides the option of using tfidf in case this is just using scripts 
    # attributes and not text embedding or vectorization
    if use_tfidf == False:
        pipe = Pipeline([('clf', classifier)])
    else:
        
        # These tfidf hyper-parameters have been tested via grid-search
        # elsewhere and found to be optimal.
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_df=.95, min_df=.1,
                                                  max_features=5000,
                                                  ngram_range=(1,2))), 
                         ('clf', classifier)])
        
#     scaler = StandardScaler
#     X_resampled_scaled = scaler.fit_transform(X_resampled)
#     X_test_scaled = scaler.transform(X_test)
    
    # Fitting the pipeline containing the tfidf processor and classifier.
    pipe.fit(X_resampled, y_resampled)
    
    # Creating predicted data.
    y_pred = pipe.predict(X_test)
    
    # Running metrics and creating a confusion matrix visual.
    print(classification_report(y_test, y_pred))

    confusion = confusion_matrix(y_test, y_pred)
    
    plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
                      show_normed=True, cmap=cmap);
    plt.show();
    
    # If there is a decision function in this classifier, we'll use it 
    # to create an ROC AUC graph.
    try:
        y_score = pipe.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print('roc_auc score: ', roc_auc)

        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show();
    except:
        pass

    return y_pred
    
#     # Creating a decision tree classifier tree graph because it's cool.
# #     if classifier == DecisionTreeClassifier():
#     dot_data = export_graphviz(classifier, out_file=None, 
# #                                    feature_names=X.columns, 
#                                class_names=np.unique(y).astype('str'), 
#                                filled=True, rounded=True, 
#                                special_characters=True)

#     # Draw graph
#     graph = graph_from_dot_data(dot_data)  
#     # Show graph
#     Image(graph.create_png()) 
#     return Image(graph.create_png()) 


def ceci_nest_pas_une_pipe(X, y, text_to_vec, classifier, 
                           cmap=plt.cm.Reds):
    """This function takes in values for a classifier and runs them through
    a pipe. Generates scores and a confusion matrix plot. As opposed with 
    the above 'script_classifiers' function, this one does not use a pipe,
    lending some additional flexibility between vectorization and modeling.
    Generally to be used with combining features and word vectors.
    
    Parameters:
        
        X: DataFrame
            Features and values to be used in model.
            
        y: Series
            Target variable for classifying.
            
        text_to_vec: word vector matrix
            In addition to features, adds a word vector matrix to join with
            the other features for modeling.
            
        Classifier: Classifier
            Which classifier to be used for training the model.
        
        cmap: pyplot color map
            Which color map to use for the confusion matrix plot."""

#     if text_to_vec:
    # Putting the TfidfVectorizer up front so I can fiddle with things
    # before the classifier.
    tfidf = TfidfVectorizer(max_df=.95, min_df=.1, max_features=5000,
                             ngram_range=(1,2))
    X2 = tfidf.fit_transform(text_to_vec)

    # Creating a sparse DataFrame to house both the features and the 
    # processed text.
    X_temp = pd.SparseDataFrame(X2, columns=tfidf.get_feature_names(),
                               default_fill_value=0)

    # Necessary for next step.
    X = X.reset_index(drop=True)

    # Combining text matrix with script attributes.
    for column in X:
        X_temp[column] = X[column]
    X = X_temp
    
    # Standard train-test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.3,
                                                    random_state=42)
    
    # There a few more bad scripts than good ones, so I'll make them even.
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(pd.DataFrame(X_train), y_train)
   
    # This is necessary for putting X into the correct format if there is only 
    # one column, as it comes out of the resampler a DataFrame when the
    # classifier actually wants a series if its only one column.
    if type(X_resampled) == pd.core.series.Series:
        X_resampled = X_resampled.iloc[:, 0]
    
    # Classifier can be whatever the user has entered as an argument. 
    clf = classifier
    clf.fit(X_resampled, y_resampled)
    
#     # The below is for creating the train scores.
#     y_train_pred = clf.predict(X_train)
    
#     # Printing out metrics and confusion matrix visual for training.
#     print(classification_report(y_train, y_train_pred))
#     print('Accuracy: ', accuracy_score(y_train, y_train_pred))

#     confusion = confusion_matrix(y_train, y_train_pred)
#     plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
#                   show_normed=True, cmap=cmap);
#     plt.show();
    
    # The below is for creating the test scores.
    y_pred = clf.predict(X_test)
    
    # Printing out metrics and confusion matrix visual for testing.
    print(classification_report(y_test, y_pred))
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    confusion = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
                      show_normed=True, cmap=cmap);
    plt.show();
    
    # If there is a decision function in this classifier, we'll use it 
    # to create an ROC AUC graph.
    try:
        y_score = classifier.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print('roc_auc score: ', roc_auc)

        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show();
    except:
        pass
    
    if feature_importance:  
        try:
            print(pd.Series(clf.feature_importances_,
                      index=X_train.columns).sort_values(ascending=False).head(200))
            df_importance = pd.Series(clf.feature_importances_, 
                                      index=X_train.columns)
            df_imp_export = df_importance.sort_values(ascending=False)
            df_importance = df_importance.sort_values(ascending=True).tail(200)
        #         df_importance.index = [labels[x] for x,y in df_importance]
            df_importance.plot(kind='barh', figsize=(8,50))
            plt.title('Most Important Features')
            plt.ylabel('Feature Name')
            plt.xlabel('Feature Importance')
            plt.show()
        except: 
            pass
    
    
#     # Creating a decision tree classifier tree graph because it's cool.
# #     if classifier == DecisionTreeClassifier():
#     dot_data = export_graphviz(classifier, out_file=None, 
# #                                    feature_names=X.columns, 
#                                class_names=np.unique(y).astype('str'), 
#                                filled=True, rounded=True, 
#                                special_characters=True)

#     # Draw graph
#     graph = graph_from_dot_data(dot_data)  
#     # Show graph
#     Image(graph.create_png()) 
#     return Image(graph.create_png()) 

def hybrid_classifier_combo(X_train, X_test, X2_train, X2_test,
                            y_train, y_test, classifier,
                            cmap=plt.cm.Reds, feature_importance=False):
    """This function takes in values for a classifier and runs them through
    a pipe. Generates scores and a confusion matrix plot. As opposed with 
    the above 'script_classifiers' function, this one does not use a pipe,
    lending some additional flexibility between vectorization and modeling.
    Generally to be used with combining features and word vectors.
    
    Parameters:
        
        X: DataFrame
            Features and values to be used in model.
            
        y: Series
            Target variable for classifying.
            
        text_to_vec: word vector matrix
            In addition to features, adds a word vector matrix to join with
            the other features for modeling.
            
        Classifier: Classifier
            Which classifier to be used for training the model.
        
        cmap: pyplot color map
            Which color map to use for the confusion matrix plot."""

#     if text_to_vec:
    # Putting the TfidfVectorizer up front so I can fiddle with things
    # before the classifier.
    tfidf = TfidfVectorizer(max_df=.95, min_df=.1, max_features=5000,
                             ngram_range=(1,2))
    X2_train = tfidf.fit_transform(X2_train)
    X2_test = tfidf.transform(X2_test)    

    # Creating a sparse DataFrame to house both the features and the 
    # processed text.
    X_temp = pd.SparseDataFrame(X2_train, columns=tfidf.get_feature_names(),
                               default_fill_value=0)
                                                        
    X_temp2 = pd.SparseDataFrame(X2_test, columns=tfidf.get_feature_names(),
                               default_fill_value=0)
    
    # Necessary for next step.
#     X_train = X_train.reset_index(drop=True)
#     X_test = X_test.reset_index(drop=True)
    
    # Combining text matrix with script attributes.
    for column in X_train:
        X_temp[column] = X_train[column]
        X_temp2[column] = X_test[column]
    X_train = X_temp
    X_test = X_temp2

    # I really wish I knew why this was necessary. I really do. But for whatever
    # reason, I keep getting just a few null values at this point in the function.
    X_test.fillna(0,inplace=True)
    X_train.fillna(0,inplace=True)
    
#     temp = pd.DataFrame(X2_train, columns=['temp'])
#     display(X_train.iloc[[40, 69, 101, 106, 147, 175, 264, 303, 343, 371, 392,
#              464, 656, 811, 963, 1099, 2024, 2044, 2066, 2265, 2554,2560]])
    
    X_train.fillna(0,inplace=True)
    # Standard train-test split.
#     X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size=.3,
#                                                     random_state=42)
    
    # There a few more bad scripts than good ones, so I'll make them even.
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(pd.DataFrame(X_train), y_train)
   
    # This is necessary for putting X into the correct format if there is only 
    # one column, as it comes out of the resampler a DataFrame when the
    # classifier actuallyl wants a series if its only one column.
#     if type(X_resampled) == pd.core.series.Series:
#         X_resampled = X_resampled.iloc[:, 0]
    
    # Classifier can be whatever the user has entered as an argument. 
    clf = classifier
    clf.fit(X_resampled, y_resampled)
    
#     # The below is for creating the train scores.
#     y_train_pred = clf.predict(X_train)
    
#     # Printing out metrics and confusion matrix visual for training.
#     print(classification_report(y_train, y_train_pred))
#     print('Accuracy: ', accuracy_score(y_train, y_train_pred))

#     confusion = confusion_matrix(y_train, y_train_pred)
#     plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
#                   show_normed=True, cmap=cmap);
#     plt.show();

#     temp = pd.DataFrame(X_test, columns=['temp'])
#     print(temp[temp['temp'].isna() == True].index)

    # temp = X_test.CCONJ.isna() == True
    # display(temp)
    
    # The below is for creating the test scores.
    y_pred = clf.predict(X_test)
    
    # Printing out metrics and confusion matrix visual for testing.
    print(classification_report(y_test, y_pred))
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    confusion = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confusion, figsize=(7,7), colorbar=True,
                      show_normed=True, cmap=cmap);
    plt.show();
    
    # If there is a decision function in this classifier, we'll use it 
    # to create an ROC AUC graph.
    try:
        y_score = classifier.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print('roc_auc score: ', roc_auc)

        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show();
    except:
        pass
    
    if feature_importance:  
        try:
            print(pd.Series(clf.feature_importances_,
                      index=X_train.columns).sort_values(ascending=False).head(200))
            df_importance = pd.Series(clf.feature_importances_, 
                                      index=X_train.columns)
            df_imp_export = df_importance.sort_values(ascending=False)
            df_importance = df_importance.sort_values(ascending=True).tail(200)
        #         df_importance.index = [labels[x] for x,y in df_importance]
            df_importance.plot(kind='barh', figsize=(8,50))
            plt.title('Most Important Features')
            plt.ylabel('Feature Name')
            plt.xlabel('Feature Importance')
            plt.show()
        except: 
            pass

def grid_search_a(X, y, classifier, param_grid, use_tfidf=True):
    """Performs a grid search to optimize parameters for classification models.
    
    Parameters:
    
        X: DataFrame
            Features and values to be used in model.
            
        y: Series
            Target variable for classifying.
            
        Classifier: Classifier
            Which classifier to be used for training the model.
        
        cmap: pyplot color map
            Which color map to use for the confusion matrix plot.
            
        use_tfidf: bool
            Whether to use tfidf on feature data prior to running model.
            
        Returns:
            dictionary of best parameters."""

    # Standard train-test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=.3,
                                                    random_state=42)
    
    # Under-sampling to even out the field.
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(pd.DataFrame(X_train), y_train)

    # If only one column, the classifier wants it as a series.
    if len(X_resampled.columns) < 2:
        X_resampled = X_resampled.iloc[:, 0]
    
    # User has a choice to use a tfidf classifier or not.
    if use_tfidf == False:
        pipe = Pipeline([('clf', classifier)])
    else:
        pipe = Pipeline([('tfidf', TfidfVectorizer(max_df=.95, min_df=.1,
                                                  max_features=5000,
                                                  ngram_range=(1,2))), 
                         ('clf', classifier)])
    
    # Performs a grid search and generates the best parameter set.
    grid_search = GridSearchCV(pipe, cv=None,
                               return_train_score=True, verbose=10,
                               scoring='accuracy', param_grid=param_grid,
                               n_jobs=-1)

    grid_search.fit(X_resampled, y_resampled)

    # Create mean training and test scores.
    training_score = np.mean(grid_search.cv_results_['mean_train_score'])
    testing_score = grid_search.score(X_test, y_test)

    # Spit out the results.
    print(f"Mean Training Score: {training_score :.2%}")
    print(f"Mean Test Score: {testing_score :.2%}")
    print("Best Parameter Combination Found During Grid Search:")
    print(grid_search.best_params_)

    # Return the parameters so they can be seen and contemplated.
    return grid_search.best_params_
            
def feature_graphics(df, col_start, col_end, add_columns=None, scatterdim=(1600,1600), 
                       figsize=(20,20), browser=False):
    """Creates scatter matrix, histograms, and corrolation matrix for 
    supplied dataframe and columns.
    
    Parameters:
        
        df: DataFrame
            DataFrame of movie information, probably from either rotten_df_cut
            or screenplays_cut.
            
        col_start: string
            Name of column in sequence to start with.
            
        col_end: string
            Name of column in sequence to end with.
            
        scatterdim: tuple
            dimensions in pixels for scatter matrix
            
        figsize: tuple
            dimensions in inches for histograms and hamburgers.
            
        browser: bool
            Indicates whether scatter matrix also opens in browser."""

    # Create subset DataFrame using start & end columns.
    temp = df[df.columns[\
        list(df.columns).index(col_start):list(df.columns)\
                                               .index(col_end)+1]].copy()
    
    # If the DataFrame is rotten_df_cut, we should add the rotten_scores column.
    temp['RottenScore'] = df.rotten_scores
    color = df.rotten_scores/100
    
    # Moving data to format Plotly will understand for scatter matrix.
    dimensions = []
    for column in temp.columns:
        dimensions.append({'label':column,'values':temp[column]})

    # Creating scatter matrix.
    data = [go.Splom(
        dimensions=dimensions,
        marker={'size':2,
                'color':color,
                'colorbar':{'thickness':20}
               }
    )]
    layout = go.Layout(height=scatterdim[1],
                       width=scatterdim[0])
    fig = go.Figure(data, layout)
    fig.show();   
    
    # Plotly can send the graph to a browser; useful for dashboards.
    if browser == True:
        pyo.plot(fig);
    
    # Create histogram of the same data.
    temp.hist(figsize=figsize);
    
    # Create corrolation heatmap of the same data.
    plt.subplots(figsize=figsize)
    sns.heatmap(temp.corr(), annot = True);
    
def stop_it(text, punct=False):
    """Removes stop words and punctuation.
    
    Arguments:
    
        text: string
            Text to have stop words and punctuation removed from..
            
        punct: bool
            If set to true, will also remove punctuation.
            
    Returns:
        String of words with stop words removed, and punctuation removed if
        selected."""
    
    text=text
    
    # Remove punctuation if indicated in arguments to do so.
    if punct == True:
        punctuations = """!()-[]{};:'"\,<>./?@#$%^&*_~"""
        for x in text: 
            if x in punctuations: 
                text = text.replace(x, '')
    
    # Split the string into a list of words to compare against the
    # spacy nlp.Defaults.stop_words list.
    split_it = text.split()
    stopped = [word.lower() for word in split_it \
               if word.lower() not in nlp.Defaults.stop_words]
    
    # Converting list back into a continuous string.
    last_word = ''
    for word in stopped:
        last_word += (' ' + word)
        
    return last_word
    
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
    ```


```python

```
