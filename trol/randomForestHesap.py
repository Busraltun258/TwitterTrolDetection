import pandas as pd
import tweepy


class randomForestHesap():

    def __init__(self):
        self.tweets = []
        self.tweetText = []

    def DownloadData(self, keyword):

        auth = tweepy.OAuthHandler('xajlUJAhcoIaelSpK6N9sA71h',
                                   'Svgrdt7cOFmpOhsWQ24wJrYGFVd4x7HuhVV5MZodZrAz7ttQF7', )

        api = tweepy.API(auth)
        limit = int(1)

        # 1 bot  0 non bot
        limit = tweepy.Cursor(api.user_timeline, screen_name=keyword, ).items(limit)
        # create DataFrame
        columns = ['screen_name', 'location', 'description', 'verified', 'follower', 'following', 'url']
        data1 = []

        for tweet in limit:
            data1.append([tweet.user.screen_name, tweet.user.location,
                          tweet.user.description, tweet.user.verified, tweet.user.followers_count,
                          tweet.user.friends_count, tweet.user.url])

        df1 = pd.DataFrame(data1, columns=columns)

        df1.to_csv('data/veri.csv')

        data1 = pd.read_csv('data/veri.csv')
        condition = (data1.screen_name.str.contains("bot", case=False) == True) | (
                data1.description.str.contains("bot", case=False) == True) | (data1.location.isnull()) | (
                            data1.verified == False)
        data1['screen_name_binary'] = (data1.screen_name.str.contains("bot", case=False) == True)
        data1['description_binary'] = (data1.description.str.contains("bot", case=False) == True)
        data1['location_binary'] = (data1.location.isnull())
        data1['verified_binary'] = (data1.verified == False)
        print(data1)
        data1.to_csv('data/sonuc.csv')
        bots = pd.read_csv('data/bots_data.csv', encoding='ISO-8859-1')
        nonbots = pd.read_csv('data/nonbots_data.csv', encoding='ISO-8859-1')

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
                            nonbots.location.isnull() == False) | (nonbots.verified == True)

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
        data_df = pd.read_csv('data/sonuc.csv', encoding=('ISO-8859-1'))
        dataset = data_df[['screen_name_binary', 'description_binary', 'location_binary', 'verified_binary']]

        print("Random Foresttahmin", model.predict(dataset))
        pred = model.predict(dataset)
        if pred == 1:
            pred = "Trol"
        else:
            pred = "Not Trol"
        print("Random Forest Classifier", pred)
        # Checking accuracy
        print("Random Forest Classifier Accuracy: {0}".format(accuracy_score(y_test, predicted)))
        follower = tweet.user.followers_count
        following = tweet.user.friends_count
        url = tweet.user.url
        name = tweet.user.name
        img = tweet.user.profile_image_url
        bg = tweet.user.profile_image_url
        return pred, keyword, follower, following, url, name, img, bg
