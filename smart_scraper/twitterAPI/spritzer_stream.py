#acknowledgment: "Introduction to Text Analysis in Python: A Hands-on Tutorial" at Summer Institute in Computational Social Science 2021.

import requests
bearer_token = "AAAAAAAAAAAAAAAAAAAAAHnOSAEAAAAAWKmBmaqipWHUCz%2BiZMIoRO%2FG1ts%3DH3ymJKodPukYp4NO6ekLqAB4NBVWoVVydMpQ3LNDgZoKg7Mrs4"
tweet_fields = ["text", "public_metrics", "created_at", "author_id", "geo"]

def tweet_stream(url, header, params):
  with requests.request("GET", url, headers=header, 
                        params=params, stream=True) as response:
    if response.status_code != 200:
        print("Got the following error: {}".format(response.text))

    else:
      for response_item in response.iter_lines():
        if response_item:
          tweet = json.loads(response_item)
          if 'includes' in tweet:
            yield {**tweet['data'], **tweet['includes']}
          else:
            yield tweet['data']

def get_streaming_tweets(bt=bearer_token,
                         tweet_fields = tweet_fields,
                         expansions = None,
                         expansion_fields = None,
                         t = None,
                         n = 100):
  
  url = "https://api.twitter.com/2/tweets/sample/stream"
  header = {"Authorization": "Bearer {}".format(bt)}
  params = {'tweet.fields': ','.join(tweet_fields)}
  if expansions:
    params['expansions'] = ','.join(expansions)
    for exp in expansion_fields:
      params[exp] = ','.join(expansion_fields[exp])
  results=[]
  if t:
    #Define the time at which you began connecting to the stream
    start = time.time()
    end = time.time()
    #Connect to the stream
    stream = tweet_stream(url, header, params)
    #Until time runs out, keep pulling tweets and append them to our receptacle
    while end-start < t:
      results.append(next(stream))
      end = time.time()
    stream.close()
  #If you passed a number of tweets to pull from the spritzer...
  elif n:
    #Connect to the stream
    stream = tweet_stream(url, header, params)
    while len(results) < n:
      results.append(next(stream))
    stream.close()
  return results

get_streaming_tweets(n=5, 
                     expansions=['author_id','geo.place_id'], 
                     expansion_fields={'user.fields': ['description','location'],
                                       'place.fields': ['country_code',
                                                        'contained_within',
                                                        'country']})
