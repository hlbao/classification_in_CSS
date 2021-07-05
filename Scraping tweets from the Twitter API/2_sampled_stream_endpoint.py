#Acknowledgement: The Summer Programs in Computational Social Science 2021.

def tweet_stream(url, header, params):
  '''
  A helper function that connects to the Twitter API's sampled stream endpoint.

  Returns a generator that yields tweets from the unfiltered spritzer stream.
  '''

  #Open the ongoing connection to the spritzer stream
  with requests.request("GET", url, headers=header, 
                        params=params, stream=True) as response:
  
    #If there was an error, print it so we know what's going on
    if response.status_code != 200:
        print("Got the following error: {}".format(response.text))

    else:

      #For each tweet that gets returned...
      for response_item in response.iter_lines():
        if response_item:
          tweet = json.loads(response_item)

          #...Pass along the tweet
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

  '''
  Function that requests tweets from the unfiltered Twitter spritzer stream.
  
  Assumes that one's Twitter dev account bearer token is stored as a string in 
  a variable called "bearer-token". 
  
  You can either pass t, the number of seconds you would like to pull tweets 
  continuously from the stream, or n, the exact number of tweets you'd like
  to pull. By default, n is set as 100.
  
  Optionally, you can also define "expansions" (list of strings), which will be
  passed to the API call, making additional information available upon request. 
  To request that information, also pass a dictionary to "expansion_fields" for 
  which its keys are the parameter names (e.g. "user.fields") and its values are
  a list of strings indicating what fields you want to request from those
  expansions.
  
  Returns a list of JSON objects, which can be traversed as a list of dictionaries.
  '''
  
  #This is the base url for the sampled stream endpoint
  url = "https://api.twitter.com/2/tweets/sample/stream"

  #Store one's bearer token in a way legible to the Twitter API
  header = {"Authorization": "Bearer {}".format(bt)}

  #Encode the desired tweet fields in a way legible to the Twitter API
  params = {'tweet.fields': ','.join(tweet_fields)}

  #If you passed expansions, include these in the API request
  if expansions:
    params['expansions'] = ','.join(expansions)
    for exp in expansion_fields:
      params[exp] = ','.join(expansion_fields[exp])

  #Create a receptacle for the API's responses
  results = []

  #If you passed a time for which to be pulling from the spritzer continuously...
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
  
  else:
    raise Exception("Must define either t or n")

  return results
