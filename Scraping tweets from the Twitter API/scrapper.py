# -*- coding:utf-8 -*-
from TwitterAPI import TwitterAPI
from TwitterAPI import TwitterPager
 
 
def search_tweets(the_consumer_key, the_consumer_secret, the_access_token_key,
                  the_access_token_secret, the_proxy_url):
    """
    Search for tweets with specific "content"
    :param the_consumer_key: the existing consumer_key
    :param the_consumer_secret: the existing consumer_secret
    :param the_access_token_key: the existing access_token_key
    :param the_access_token_secret: the existing access_token_secret
    :param the_proxy_url: proxy and port number
    :return:
    """
    api = TwitterAPI(consumer_key=the_consumer_key,
                     consumer_secret=the_consumer_secret,
                     access_token_key=the_access_token_key,
                     access_token_secret=the_access_token_secret,
                     proxy_url=the_proxy_url)
    r = TwitterPager(api, 'search/tweets', {'q': 'pizza', 'count': 10})
    for item in r.get_iterator():
        if 'text' in item:
            print item['text']
        elif 'message' in item and item['code'] == 88:
            print 'SUSPEND, RATE LIMIT EXCEEDED: %s\n' % item['message']
            break
 
if __name__ == "__main__":
    consumerKey = ""  #your own key
    consumerSecret = ""
    accessToken = ""
    accessTokenSecret = ""
    
    search_tweets(the_consumer_key=consumerKey,
                  the_consumer_secret=consumerSecret,
                  the_access_token_key=accessToken,
                  the_access_token_secret=accessTokenSecret)
