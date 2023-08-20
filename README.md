# IBM-article-recommender

<br>
<div align="center">
<img src="https://ibm.github.io/watson-studio-workshop/housing-price-predictor/assets/watson_logo.png">
</div>
<br>

### Table of Contents

1. [Installation](#installation)
2. [Project Description](#project-description)
    - [Project motivation](#motivation)
    - [Objectives](#objective)
3. [File Structure](#files)
4. [How to interact](#interact)
5. [Suggestion](#suggestion) 
6. [Licensing, Authors, and Acknowledgements](#licensing)


## 1. Installation <a name="installation"></a>
* No need to install libraries beyond the Anaconda distribution of Python
* The code should run with no issues using Python versions 3.*. Currently using `Python 3.11.3` on `Windowns 11`.

## 2. Project Description <a name="project-description"></a>

### 2.1. Project motivation <a name="motivation"></a>
In the IBM Watson Studio, there is a large collaborative community ecosystem of articles, datasets, notebooks, and other A.I. and ML. assets. Users of the system interact with all of this.

This project will analyze the interactions that users have with articles on the IBM Watson Studio platform, and make recommendations to them about new articles they may like.

### 2.2. Objectives <a name="objective"></a>
1. Apply `user-user based collaborative filtering` method to make recommendation. By looking at users that are similar in terms of the items they have interacted with. These items would then be recommended to the similar users.
2. Apply `rank-based` method to give recommendation for new users by finding the most popular articles simply based on the most interactions.

**Note**: new users mean users without any interaction information so that we can not apply `user-user based collaborative filtering` method to make recommendation (aka `cold start problem`)

3. Blend the code into a class that can be easily used with a web application ([recommender.py](https://github.com/hongtranthianh/IBM-article-recommender/blob/main/recommender.py)).


## 3. File Structure<a name="files"></a>

```

- data
|- articles_community.csv  # Article information on IBM platform
|- user-item-interactions.csv  # Interaction information about users and articles 
                               # (only use this data in this project)

- Preparation # Folder contains all preparation files before blending the code into a class

- EDA.ipynb # Get to know the data

- recommender_functions.py # Functions facilitate recommendations

- recommender_template.py # Suggested way to structure recommender class 

- recommender.py # The recommender class that can be easily used with a web application

- README.md

```

## 4. How to interact<a name="interact"></a>
**Step 1**: Pull this repository to your local machine
```
git clone https://github.com/hongtranthianh/IBM-article-recommender.git
```

**Step 2**: Run these `.py` files in the project root directory 

```
py recommender_functions.py
```

```
py recommender.py
```

## 5. Suggestion<a name="suggestion"></a>
This project can be further developed in several ways:
1. Move the code in a web application (e.g., `flask app`) to show off the result
2. Package the recommendation engine code to be pip installed by others
3. Apply `content based recommendations` method to provide recommendations using NLP skill. By categorizing articles based on information about the content of articles in [articles_community.csv](https://github.com/hongtranthianh/IBM-article-recommender/blob/main/data/articles_community.csv), then put it into `rank-based recommendation` and finally use it as filter for `user-user based collaborative filtering`.

## 6. Licensing, Authors, and Acknowledgements<a name="licensing"></a>
This project is part of the Data Scientist Nanodegree program from Udacity, under module *Experimental Design & Recommendations*.

Thanks Udacity for providing a guideline for this project and thanks IBM for providing the datasets.