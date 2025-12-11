Sentiment analysis of Game of Thrones 

datasets from Kaggle

Goal:
To determine what emotional tone, key themes, and keywords correlate with the highest-rated TV show episodes — based on scripts and ratings.

After cleaning the data, a linear regression model will be used to understand the correlation between combinations of emotions and episode ratings.

Notes

There are some 8 thousand missing speaker values, they include descriptions of the scene's direction and things like the visuals and actions of crowds, these were given the value of Narration


The target value for the initial regression model was the IMDb Rating. There is very little variance in the scores for Game of Thrones, which caused a low R^2 score, prompting me to test all of the other numerical columns.


U.S. Viewers, running time, and metacritic ratings proved to have the highest r^2 values / were the best target variables to compare the impact of the emotional coefficients with.  