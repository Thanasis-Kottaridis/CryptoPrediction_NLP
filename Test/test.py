import praw
from pmaw import PushshiftAPI

# REDDIT AUTH
CLIENT_ID = "IcTrWsQDFCcZEe3rWrlB4A"

SECRET_KEY = 'HZQy-nneDNv4THu_G8MhVJ96KOq4cg'

if __name__ == '__main__':
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=SECRET_KEY,
        user_agent='MyBot/0.0.1'
    )

    api_praw = PushshiftAPI(praw=reddit)
    api = PushshiftAPI()
    posts = api.search_submissions(
        before=1672915636,
        after=1635527249,
        subreddit="Bitcoin",
        sort="desc",
        limit=100
    )
    post_list = [post for post in posts]
    print(post_list)