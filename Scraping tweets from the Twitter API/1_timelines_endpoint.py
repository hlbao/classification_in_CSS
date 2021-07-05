bearer_token = "**************** INSERT YOUR BEARER TOKEN HERE ****************"

#tweet_fields = ["text", "public_metrics", "created_at", "author_id", "geo"]
tweet_fields = ["text", "public_metrics", "created_at", "author_id", "geo", "attachments", "context_annotations",
                "conversation_id", "entities", "lang", "possibly_sensitive"]

def get_user_tweets(users, 
                    bt=bearer_token,
                    tweet_fields = tweet_fields,
                    expansions = None,
                    expansion_fields = None,
                    start_time = None, 
                    end_time = None):
  
  #create a receptacle for the API's responses
  responses = []

  for user in users:

    url = "https://api.twitter.com/2/users/{}/tweets".format(user)

    #Store one's bearer token in a way legible to the Twitter API
    header = {"Authorization": "Bearer {}".format(bt)}

    #Encode the desired tweet fields in a way legible to the Twitter API
    params = {"tweet.fields": ','.join(tweet_fields)}

    #If you passed expansions, include these in the API request
    if expansions:
      params['expansions'] = ','.join(expansions)
      for exp in expansion_fields:
        params[exp] = ','.join(expansion_fields[exp])

    #If you defined a start and/or end time, add them to the API request
    if start_time:
      params['start_time'] = start_time
    if end_time:
      params['end_time'] = end_time

    #Make the actual call to the Twitter API
    response = requests.request("GET", url, headers=header, params=params)

    #If there's an error, be sure to print it out so we know what's going on
    if response.status_code != 200:
      print("Got the following error for user {0}: {1}".format(user, response.text))

    #Add the user's tweets to our receptacle
    responses.append(response.json())

  return responses

#The user IDs of my Twitter account
users = ["1240776894288728069"]

results = get_user_tweets(users, start_time="2021-06-01T00:00:01Z",
                          expansions=["author_id"],
                          expansion_fields={"user.fields": ["description"]})

#Print the most recent tweet from my account
results[0]['data'][0]
#Print my user information
results[0]['includes']

