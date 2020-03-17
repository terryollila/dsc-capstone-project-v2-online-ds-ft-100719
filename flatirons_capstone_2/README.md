# What's a Writer Worth?
## Screenplay Science and the Value of a Few Good Words

This project is an attempt to use machine learning methods to predict whether or not a film will be highly rated by critics based on the language of the screenplay.

Methods employed were 

- Web scraping of multiple sites, using the beautifulsoup Python library.
- Natural language processing, especially TFIDF vectorization and use of the spacy and nltk Python libraries to dissect the text.
- Numerous classification models, including but not limited to random forests, support vector machines, and neural networks.
- Matplotlib for various inline graphics.
- Web-based dashboard creation using plotly, dash, and wordcloud.
- Latent Derichlet allocation for grouping films by language patterns.

First, some important files for navigating this repo:

1. main.ipynb. This is the main technical notebook, which is a summary view of the 5 separate notebooks that make up this project. For the sake of size, some things have been kept out of this file, such as most of the models that weren't as strong as others, and the code for creating the dashboard and visualizations. For all practical purposes, this can be considered the main notebook file, and the others supplementary, except perhaps for visuals.ipynb and functions.py.

2. obtain.ipynb. This is the main file used for web scraping the necessary information from rottentomatoes, metacritic, and springfieldspringfield.

3. scrubbing.ipynb. This is where the data from the sites listed above is cleaned to ensure integrity down the line.

4. eda.ipynb. An important part of this project, this is the exploratory data analysis file, where I did all of the work using nlp to dissect the screenplays.

5. modeling.ipynb. A large file, this is where I used roughly 30 models in attempting to find the best means of predicting screenplay value.

6. visuals.ipynb. This is where I did all the work creating plotly graphs and embedding them into html using the python dash library.

7. functions.py. This is where the various functions were kept, as they eventually took up a large portion of the notebook.



Further links to come include the site for the interactive dashboard, and the blog post written about this project.


## Abstract

The film industry worldwide does upwards of 50 billion dollars in box office sales, not counting home entertainment revenue, which brings it up closer to 150 billion dollars. Operating within that brick of cash comes with a tremendous amount of risk, with films from major studios sometimes spending a quarter of a billion dollars or more on a single film. Decisions made at smaller studios are no less important to them, as they might be putting their entire livelihoods on the line in the hopes of a hit. And the value of a movie begins with a script.

My premise in creating this project was to ascertain whether a movie's critical rating can be determined to any extent by the text of its screenplay alone. Before attaching a cast and director and crew and all of the other costs associated with creating a cinematic work, having some guidance as to the quality of the script itself can be a benefit in minimizing risk. While an algorithm is no substitute for having a human eye on a screenplay, some level of unbiased machine learning can be leveraged to take a closer look. It is also possible to use this process to vet possible licensing if resources are short and slush piles are large.

As there are many factors going into a movie's rating, such as cast, director, editing, music, costuming, set design and so on, it is not necessarily expected that a movies critical rating can be determined solely by the text of its screenplay. However, there is still much value to be had if any measureable predictability can be found. At the completion of my modeling, I was ultimately able to predict scripts from good movies and bad movies, as rated by metacritic.com, about 65% of the time. Given the other factors in rating, I feel that is a significant enough to create value in the model. Among the various models I tried, some had a better true positive rate, and others had a better false positive rate, so there are some choices there depending on what use cases might be found, such as when it might be more advantageous to find a good movie versus avoiding a bad one.

Recommendations to a given filmmaker would be to use modeling to sort potential screenplays into lists of scripts with higher likelihood of success, using modeling to evaluate scripts in process and step back to consider if it needs more work if the model doesn't like it, and for the screenwriters themselves, to check their scripts against the model and if it comes back with a 'bad' rating, potentially rethink their life choices.

For further research, I would like to create a text ingestion field in the dashboard that allows a user to insert a body of text and have a prediction returned evaluating the content as a screenplay and assigning a good or bad designation. I would create predictability functionality that would allow a user to choose a 'good' or 'bad' setting and have automatically generated text returned back in the style of either a good or bad screenplay. And I would go deeper into the neural networds when modeling, especially toward regression. They showed promise when using them in this project, but there was insufficient time to build them out using pre-made embedding layers and so on.



# Obtaining Data

All data for this report was gathered using the following three sites:
    
- metacritic.com for movie rating information.
- rottontomatoes.com for additional movie rating information.
- SpringfieldSpringfield.co.uk for gathering the screenplay texts.

I the most highly rated and most lowly rated films as listed on metacritic. These extremes were be used for training my classification models to pridict if random movies will be highly rated or lowly rated films.

Knowing that I won't be able to come up with screenplays for every single movie, I took 2000 great and 2000 terrible films, in the hopes of winding up with at least 1000 of each.


The rottentomatoes.com information was used for linear regression. Whereas with metacritic we were using only the best and worst for classification, here I was using samples from the entire spectrum for regression analysis.


As this is an analysis centered around natural language processing, my primary data source is he screenplay text from every movie.

# Scrubbing Data

In the scrubbing process, I created some script metrics in order to get a sense of how the good scripts might fundamentally differ from the bad.

Some of those metrics were:


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


# Exploratory Data Analysis

This is where a large bulk of the work was performed.

I broke apart the scripts into tokens and used the nlp libraries to score sentiment and separate parts of speech, both coarse and fine. Additional metrics such as word count were generated at this point.

I used latent Derichlet allocation as an unsupervised method to group the films into clusters with similar language patters. This created some startlingly accurate groupings of similar films, with a few humorous exceptions.

Scatter matrices, histograms, and correlation heat maps were generated for all features in order to get a sense of correlation, normality, and linearity. 

Below are the top words for the categories generated by LDA:

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
    ['gun', 'brother', 'goddamn', 'motherf**ker', 'dude', 'f**ked', 'em', 'yo', 'jesus', 'b***h', 'a**', 'f**kin', 'aint', 'f**king', 'f**k']
    
    
    THE TOP 15 WORDS FOR CATEGORY #7
    ['buy', 'women', 'parents', 'york', 'bye', 'perfect', 'dinner', 'book', 'party', 'honey', 'movie', 'girls', 'sex', 'married', 'mom']
    
    
    THE TOP 15 WORDS FOR CATEGORY #8
    ['white', 'women', 'ii', 'sort', 'mmm', 'mm', 'wow', 'alright', 'mmhmm', 'john', 'bob', 'hmm', 'ah', 'um', 'uh']
    
    
    THE TOP 15 WORDS FOR CATEGORY #9
    ['groaning', 'breathing', 'chuckles', 'continues', 'music', 'groans', 'screams', 'panting', 'mary', 'sighs', 'michael', 'grunting', 'grunts', 'gasps', 'screaming']
    

# Modeling

For modeling, I used the extreme data from metacritic to train the models, then the 0-100 data from rottentomatoes for testing.

My first set of models employ TFIDF Vectorization using a variety of classifiers. I then go into using only the engineered features such as parts of speech and word count. From there, I merge the TFIDF matrix with the features. Finally, I use a support vector classifier with TFIDF, then feed the results into a random forest model using engineered features. This produced the best result, with 68% accuracy, 64% true positive rate, and 70% true negative rate.

I use support vector classifiers, decision trees, logistic regression, random forests, XG Boost, Naive-Bayes, and neural networks.

I followed this up with a regression analysis using only the rottentomatoes data, but those results were negligible with about a .1 r^2 test score.

# Visualization

For bringing the statistical information about the screenplays to life, I used an html dashboard that I coded using the plotly, dash, and wordcloud libraries in Python.
A word cloud showed the top words from each side that were mutually exclusive to each other:

![thumbs up/down word cloud](https://raw.githubusercontent.com/terryollila/dsc-capstone-project-v2-online-ds-ft-100719/master/flatirons_capstone_2/images/two_thumbs-md.png)

I created a horizontal bar chart to display some of the words that had overall importance in the modeling, which switched between the three by means of radio buttons, and to demonstrate the correlation for each.

![important features](https://raw.githubusercontent.com/terryollila/dsc-capstone-project-v2-online-ds-ft-100719/master/flatirons_capstone_2/images/all_words.png)

And I used an interactive dropdown to show numerous features such as unique words, sentiment, and parts of speech. Here is an example:

![word feature distribution plots](https://raw.githubusercontent.com/terryollila/dsc-capstone-project-v2-online-ds-ft-100719/master/flatirons_capstone_2/images/dist_example.png)

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
