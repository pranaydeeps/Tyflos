import tweepy

langs = ['en','nl','hi','es','pt','tgl','bn','de','fr','id','km','ms','my','ro','th','vi','zh']

consumer_key = 'LktBSkcbJ0bEiwuDBCk0retrb'
consumer_secret = 'X6qcAQpJmqCbzHKnH5CfQphDm3H1eiwpirA7qXw3OZmZrH9yM7'

access_token = '915011546081841152-15NWOZi5zSfi6VdtjibtseUmF7bMk6C'
access_token_secret = 'zEEXJPo806NxOhkmJ2BrJoJnkWNuiKraUKFaPDTv5r7v6'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

print("Twitter Auth Successful. Proceeding with Scraping.")

def get_full_text(id):
    tweet = api.get_status(id, tweet_mode='extended')
    print(tweet.full_text)

with open('hashtags_shortened.txt') as f:
    hashtags = f.read().splitlines()

for i in langs:
    for j in hashtags:
        current_data = []
        search_query = '{} lang:{}'.format(j, i)
        tweets = api.search_full_archive('Tyflos',search_query)
        for tweet in tweets:
            full_text = get_full_text(tweet.id)
            if full_text.startswith('RT'):
                continue
            if len(full_text.split()<10):
                continue
            current_data.append(get_full_text(tweet.id))
        while(len(current_data)<2000):
            tweets = api.search_full_archive('Tyflos',search_query, next)
            for tweet in tweets:
                current_data.append(get_full_text(tweet.id))
        