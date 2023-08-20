import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments

class Recommender():
    '''
    Apply rank_based and user-user collaborative methods to make recommendation about articles that users may be of interest.
    '''

    def __init__(self):
        '''
        I didn't have any required attributes needed when creating my class.
        '''

    def make_recommendations(self, user_item_interactions_pth, user_id, m=10):
        '''
        INPUT:
        user_item_interactions_pth - file path
        user_id - (int) a user id
        m - (int) the number of recommendations you want for the user

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title
        '''
        # Store inputs as attributes
        self.interaction_df = pd.read_csv(user_item_interactions_pth)
        # Set up useful values to be used through the rest of the function
        self.interaction_email_mapper_df = rf.email_mapper_df(self.interaction_df)
        self.user_item_df = rf.create_user_item_matrix(self.interaction_email_mapper_df)
        self.user_ids_series = np.array(self.user_item_df.index)
        self.n_users = self.user_item_df.shape[0]

        if user_id in self.user_ids_series:
            recs, rec_names = rf.user_user_recs_part2(user_id, self.interaction_email_mapper_df, m)
        else:
            # For the cold start problem
            recs = rf.get_top_article_ids(m, self.interaction_email_mapper_df)
            rec_names = rf.get_top_articles(m, self.interaction_email_mapper_df)
            print("Because this user wasn't in our database, we are giving back the most popular article recommendation")

        return recs, rec_names
    
if __name__ == '__main__':
    import recommender as r

    #instantiate recommender
    rec = r.Recommender()

    m=10
    user_id_1 = 5
    user_id_2 = 12000

    # make recommendations
    print('Top {} recs for user_id = {}'.format(m,user_id_1))
    print(rec.make_recommendations(user_item_interactions_pth='data/user-item-interactions.csv',user_id=user_id_1, m=m)) # user in the dataset
    print()
    print('Top {} recs for user_id = {}'.format(m,user_id_2))
    print(rec.make_recommendations(user_item_interactions_pth='data/user-item-interactions.csv',user_id=user_id_2, m=m)) # user not in dataset
    print()
    print('There are {} users in the dataset'.format(rec.n_users))

    
