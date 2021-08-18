#acknowledgment: "Introduction to Text Analysis in Python: A Hands-on Tutorial" at Summer Institute in Computational Social Science 2021.

import requests
pd.options.mode.chained_assignment = None

bearer_token = "AAAAAAAAAAAAAAAAAAAAAHnOSAEAAAAAWKmBmaqipWHUCz%2BiZMIoRO%2FG1ts%3DH3ymJKodPukYp4NO6ekLqAB4NBVWoVVydMpQ3LNDgZoKg7Mrs4"
tweet_fields = ["text", "public_metrics", "created_at", "author_id", "geo"]

def get_user_tweets(users, 
                    bt=bearer_token,
                    tweet_fields = tweet_fields,
                    expansions = None,
                    expansion_fields = None,
                    start_time = None, 
                    end_time = None):
  responses = []
  for user in users:
    url = "https://api.twitter.com/2/users/{}/tweets".format(user)
    header = {"Authorization": "Bearer {}".format(bt)}
    params = {"tweet.fields": ','.join(tweet_fields)}
    if expansions:
      params['expansions'] = ','.join(expansions)
      for exp in expansion_fields:
        params[exp] = ','.join(expansion_fields[exp])
    if start_time:
      params['start_time'] = start_time
    if end_time:
      params['end_time'] = end_time
    response = requests.request("GET", url, headers=header, params=params)
    if response.status_code != 200:
      print("Got the following error for user {0}: {1}".format(user, response.text))
    responses.append(response.json())
  return responses

#The user IDs of Winson's and my Twitter accounts
users = ["440645901", "1240776894288728069"]
results = get_user_tweets(users, start_time="2021-06-01T00:00:01Z",
                          expansions=["author_id"],
                          expansion_fields={"user.fields": ["description"]})

results[1]['data']
results[0]['includes']
