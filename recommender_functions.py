import pandas as pd
import numpy as np

# Part 1: Rank-Based Recommendations
def get_top_articles(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    '''
    top_articles = df['title'].value_counts().index[:n].tolist()
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    '''
    top_articles = df['article_id'].value_counts().index[:n].tolist()
 
    return top_articles # Return the top article ids


# Part 2: User-User Based Collaborative Filtering
def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    user_item_df = df.groupby(['user_id','article_id']).count().reset_index()
    user_item_df.rename(columns={'title':'count'}, inplace=True)
    user_item_df['count']=1
    user_item = user_item_df.groupby(['user_id','article_id'])['count'].max().unstack().fillna(0)
    
    return user_item # return the user_item matrix 

def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    article_names = df[df['article_id'].isin(article_ids)]['title'].unique().tolist()
    
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    article_ids = user_item.loc[user_id,:].loc[lambda x: x==1].index.tolist()
    article_names = get_article_names(article_ids,df)
    
    return article_ids, article_names # return the ids and names

def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # Measure the similarity of each user to the provided user_id
    similarity_df = user_item.dot(np.transpose(user_item)).loc[user_id,:].reset_index()
    similarity_df.columns = ['user_id', 'similarity']

    # Count the number of articles viewed by each user
    interaction_df = df.groupby('user_id')['article_id'].count().reset_index()
    interaction_df.columns = ['user_id','num_interactions']
    
    # Merge similarity_df and interaction_df
    neighbors_df = pd.merge(similarity_df, interaction_df, 'left', 'user_id')
    # Sort the neighbors_df by the similarity and then by number of interactions
    neighbors_df = neighbors_df.sort_values(['similarity','num_interactions'], ascending=False)
    # Rename column
    neighbors_df.rename(columns = {'user_id':'neighbor_id'}, inplace=True)
    # Remove the own user's id
    neighbors_df = neighbors_df[neighbors_df['neighbor_id'] != user_id]
    
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    # Initiate an empty numpy array to store recommendation
    recs = np.array([])

    # Find similar user ids to the input user_id and the articles was seen by the user
    similar_user_ids = get_top_sorted_users(user_id, df=df, user_item=user_item)['neighbor_id'].tolist()
    seen_articles = get_user_articles(user_id, user_item)[0]

    # Loops through the users based on closeness to the input user_id
    for similar_user_id in similar_user_ids:
        seen_article_of_similar_user = get_user_articles(similar_user_id, user_item)[0]
        # Finds articles the user hasn't seen before and provides them as recommendation
        rec_articles = np.setdiff1d(seen_article_of_similar_user, seen_articles, assume_unique=True)
        # Update 'recs' with the above 'rec_articles'
        recs = np.unique(np.concatenate([rec_articles, recs], axis=0))
        # If the number of current recommendations exceed given m, break the for loop
        if len(recs) >= m:
            break
    # Limit to m recommendation 
    recs = recs[:m].tolist()

    # Get article name
    rec_names = get_article_names(recs,df)
    
    return recs, rec_names

# Part 3: Content Based Recommendations (EXTRA - NOT REQUIRED)
def make_content_recs():
    '''
    INPUT:
    
    OUTPUT:
    
    ''' 

