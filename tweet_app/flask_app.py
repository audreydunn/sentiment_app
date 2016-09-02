from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import os
import numpy as np
from vectorizer import vect
from vectorizer import tokenizer
import tweepy

app = Flask(__name__)
consumer_key = "eImLPl37S2Q9utJhjYGxQqAuF"
consumer_secret = "6bBSgJkWnDISa0CqxSdZk6OBNZ0oa8QDJL4cKTE8Ncq59KgDTi"
access_key = "741408750460936194-rpyOyhhoFzcx0zecz8pWv3QIpwknzbE"
access_secret = "LkVdcofeR7uA9gqVro4xgtDPImFFQ6E5TKJC05tHRUiP6"

# Preparing classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects/classifier.pkl'), 'rb'))

def classify(document):
    label = {0: 'positive', 1: 'negative'}
    proba_list = []
    neg_match = 0
    pos_match = 0
    percent = 0
    for i in document:
        new = tokenizer(i)
        # print(new)
        x = vect.transform(new)
        X = x.toarray()
        # print(X)
        y = clf.predict(X)
        # print(y)
        proba = clf.predict_proba(X).max()
        if y == 1:
            neg_match += 1
        else:
            pos_match += 1
        if neg_match >= pos_match:
            y_final = 1
        else:
            y_final = 0
        proba_list.append(proba)
    proba_final = sum(proba_list) / float(len(proba_list))
    if label[y_final] == 'positive':
        percent += pos_match / (pos_match + neg_match)
    elif label[y_final] == 'negative':
        percent += neg_match / (pos_match + neg_match)
    return label[y_final], proba_final, percent

def get_all_tweets(screen_name):
    # Twitter only allows access to a users most recent 3240 tweets with this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    alltweets = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:

        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1


    # transform the tweepy tweets into a 2D array that will populate the csv
    outtweets = [tweet.text.encode("utf-8") for tweet in alltweets]
    return outtweets


class TweetForm(Form):
    tweetreview = TextAreaField('', [validators.DataRequired(), validators.length(min=4)])

@app.route('/')
def index():
    form = TweetForm(request.form)
    return render_template('tweetform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = TweetForm(request.form)
    if request.method == 'POST' and form.validate():
        username = request.form['tweetreview']
        text = get_all_tweets(username)
        y, proba, p_value = classify(text)
        return render_template('results.html', content='Placeholder', prediction=y, probability=round(proba*100, 2),
                               p_final=round(p_value*100, 2))

    return render_template('tweetform.html', form=form)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tweetform')
def tweetform():
    form = TweetForm(request.form)
    return render_template('tweetform.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
