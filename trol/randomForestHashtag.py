# Python Script to Extract tweets of a
# particular Hashtag using Tweepy and Pandas
import numpy as np
# import modules
import pandas as pd
import tweepy


class randomForestHashtag():
    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def scrape(self, words):

        # Enter your own credentials obtained
        # from your developer account
        consumer_key = "xajlUJAhcoIaelSpK6N9sA71h"
        consumer_secret = "Svgrdt7cOFmpOhsWQ24wJrYGFVd4x7HuhVV5MZodZrAz7ttQF7"

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        api = tweepy.API(auth)

        # Enter Hashtag and initial date
        print("Enter Twitter HashTag to search for")

        # number of tweets you want to extract in one run
        numtweet = int(1000)
        # Creating DataFrame using pandas
        db = pd.DataFrame(columns=['username',
                                   'description',
                                   'location',
                                   'following',
                                   'followers',
                                   'totaltweets',
                                   'retweetcount',
                                   'text',
                                   'hashtags',
                                   'verified',
                                   'url'])
        # restricted using .items(number of tweets)
        tweets = tweepy.Cursor(api.search_tweets, words, lang="tr", tweet_mode='extended').items(numtweet)

        list_tweets = [tweet for tweet in tweets]

        def printtweetdata(n, ith_tweet):
            print()
            print(f"Tweet {n}:")
            print(f"Username:{ith_tweet[0]}")
            print(f"Description:{ith_tweet[1]}")
            print(f"Location:{ith_tweet[2]}")
            print(f"Following Count:{ith_tweet[3]}")
            print(f"Follower Count:{ith_tweet[4]}")
            print(f"Total Tweets:{ith_tweet[5]}")
            print(f"Retweet Count:{ith_tweet[6]}")
            print(f"Tweet Text:{ith_tweet[7]}")
            print(f"Hashtags Used:{ith_tweet[8]}")
            print(f"verified:{ith_tweet[9]}")
            print(f"url:{ith_tweet[10]}")

        # Counter to maintain Tweet Count
        i = 1

        # we will iterate over each tweet in the
        # list for extracting information about each tweet
        for tweet in list_tweets:
            name = tweet.user.screen_name
            description = tweet.user.description
            location = tweet.user.location
            following = tweet.user.friends_count
            follower = tweet.user.followers_count
            totaltweets = tweet.user.statuses_count
            retweetcount = tweet.retweet_count
            hashtags = tweet.entities['hashtags']
            verified = tweet.user.verified
            url = tweet.user.url
            img = tweet.user.profile_image_url

            # Retweets can be distinguished by
            # a retweeted_status attribute,
            # in case it is an invalid reference,
            # except block will be executed
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:
                text = tweet.full_text
            hashtext = list()
            for j in range(0, len(hashtags)):
                hashtext.append(hashtags[j]['text'])

            # Here we are appending all the
            # extracted information in the DataFrame
            ith_tweet = [name, description,
                         location, following,
                         follower, totaltweets,
                         retweetcount, text, hashtext, verified, url]
            db.loc[len(db)] = ith_tweet

            # Function call to print tweet data on screen
            printtweetdata(i, ith_tweet)
            i = i + 1
        filename = 'data/scraped_tweets.csv'

        # we will save our database as a CSV file.
        db.to_csv(filename)
        print('Scraping has completed!')
        data1 = []
        data1 = pd.read_csv('data/scraped_tweets.csv')
        condition = (data1.username.str.contains("bot", case=False) == True) | (
                data1.description.str.contains("bot", case=False) == True) | (data1.location.isnull()) | (
                            data1.verified == False)
        data1['screen_name_binary'] = (data1.username.str.contains("bot", case=False) == True)
        data1['description_binary'] = (data1.description.str.contains("bot", case=False) == True)
        data1['location_binary'] = (data1.location.isnull())
        data1['verified_binary'] = (data1.verified == False)
        print(data1)
        data1.to_csv('data/scraped_tweets.csv')
        bots = pd.read_csv('data/bots_data.csv', encoding=('ISO-8859-1'))
        nonbots = pd.read_csv('data/nonbots_data.csv', encoding=('ISO-8859-1'))
        # Creating Bots identifying condition
        # bots[bots.listedcount>10000]
        condition = (bots.screen_name.str.contains("bot", case=False) == True) | (
                bots.description.str.contains("bot", case=False) == True) | (bots.location.isnull()) | (
                            bots.verified == False)

        bots['screen_name_binary'] = (bots.screen_name.str.contains("bot", case=False) == True)
        bots['description_binary'] = (bots.description.str.contains("bot", case=False) == True)
        bots['location_binary'] = (bots.location.isnull())
        bots['verified_binary'] = (bots.verified == False)
        print("Bots shape: {0}".format(bots.shape))

        # Creating NonBots identifying condition
        condition = (nonbots.screen_name.str.contains("bot", case=False) == False) | (
                nonbots.description.str.contains("bot", case=False) == False) | (
                            nonbots.location.isnull() == False) | (
                            nonbots.verified == True)

        nonbots['screen_name_binary'] = (nonbots.screen_name.str.contains("bot", case=False) == False)
        nonbots['description_binary'] = (nonbots.description.str.contains("bot", case=False) == False)
        nonbots['location_binary'] = (nonbots.location.isnull() == False)
        nonbots['verified_binary'] = (nonbots.verified == True)
        print("Nonbots shape: {0}".format(nonbots.shape))

        # Joining Bots and NonBots dataframes
        df = pd.concat([bots, nonbots])
        print("DataFrames created...")
        # Splitting data randombly into train_df and test_df
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(df, test_size=0.2)
        print("Randomly splitting the dataset into training and test, and training classifiers...\n")

        # Using Random Forest Classifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        clf = RandomForestClassifier(min_samples_split=50, min_samples_leaf=200)

        # 80%
        X_train = train_df[
            ['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']]  # train_data
        y_train = train_df['bot']  # train_target

        # 20%
        X_test = test_df[
            ['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']]  # test_Data
        y_test = test_df['bot']  # test_target

        # Training on decision tree classifier
        model = clf.fit(X_train, y_train)

        # Predicting on test data
        predicted = model.predict(X_test)
        dataset = data1[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']]
        tahmin = model.predict(dataset)
        dfr = pd.read_csv('data/scraped_tweets.csv')
        dfr['troldurum'] = tahmin
        dfr.to_csv('data/sonsonuc.csv', index=None)
        name = tweet.user.name
        return words, name
