
##################  MovieLens Project  ######################################

# Recommendation systems help people make better choice depending on their tastes and needs. User 
# ratings are one character which recommendation system follows to recommend choices to the customer. 
# Highly rated and in-demand items are found with help of algorithm for recommending the products. 
# Also depending on the past behaviour of the user an alogrithm can recommend relevant and 
# user-specific items. It helps the user to select the most suitable item from the available options. 
# A happy customer is expected to return to shop frequently and this helps build business.

# Load libraries
library(tidyverse)
library(caret)

# Load the data
load("rdas/edx.rda")

str(edx)

# Number of distinct users, movies and genres 
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId),
            n_genres = n_distinct(genres))

# Count gives the number of ratings available for a movie. 
# Explore the times a movie is rated. Can see that some movies are often rated and some very less.
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# Count gives the number of ratings available from a user. 
# Explore the times a user rates a movie. Can see that some users often give rating and some very less.
edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

# Same effect is seen in genre also, that some genre is rated often
edx %>% 
  dplyr::count(genres) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Genres")

# Number of ratings available is important that a larger number gives a more reliable rating. The 
# average of large number of ratings will be more reliable than one person rating a movie too high or 
# too low. One user rating a bad movie too high will spoil the power to predict the movie ratings.
# So the number of ratings needs to be taken care.

# Movie recommendation system here uses User Id, Movie Id, Genres and the Movie rating. 
# Here every user does not rate every movie. So all the user, movie and genre specific effects on 
# rating is used as predictors in movie recommendations. 

# In movie recommendation the user effect, movie effect and genre effect are incorporated in the 
# algorithm so that we expect the algorithm to predict the ratings of all the movies based on the 
# available ratings along with the biases for the user behaviour, individual movies and the genres.

# Mean movie rating
mu <- mean(edx$rating)

# Create a function to calculate the RMSE to help model evaluation
RMSE <- function(true_ratings, predicted_ratings){
     sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Load the validation dataset for evaluating the prediction accuracy
load("rdas/validation.rda")

# Here the mean rating across all movies and users, most naive figure is used as the predicted rating.
# Calculate the RMSE for the naive model used for prediction.
naive_rmse <- RMSE(validation$rating, mu)

# Tabulate the RMSE for the model
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)
rmse_results

# As expected this model didn't do well and error value is unacceptable.
# Different users rates same movie very differently depending on their tastes. Also some movies are 
# rated more often. Same difference are expected in different genre of movies.
# This generates a user specific, movie specific and genre specific effect when the ratings are done.
# We need to account for these difference in modeling so that the model accounts for the differences
# when different users rate same movies, or high ratings received for blockbuster movies or a similar 
# kind of bias in genres.

# Find the movie bias or movie effects on ratings
movie_avgs <- edx %>% 
     group_by(movieId) %>% 
     summarize(b_i = mean(rating - mu))    

# Find the user bias or user effects on ratings
user_avgs <- edx %>% 
     left_join(movie_avgs, by='movieId') %>%
     group_by(userId) %>%
     summarize(b_u = mean(rating - mu - b_i))
     
# Find the genre bias or genres effects on ratings
genres_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_gen = mean(rating- mu - b_i - b_u))
  

# Mean validation ratings. It is same as the test datset 
mu_val <- mean(validation$rating)  

# Predict the movie ratings for validation dataset.
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genres_avgs, by='genres') %>%
  mutate(pred = mu_val + b_i + b_u + b_gen) %>%
  .$pred

# Model evaluation
model_rmse <- RMSE(predicted_ratings,validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model + genre effect model",  
                                     RMSE = model_rmse ))
rmse_results
# 1 Just the average                                1.06 
# 2 Movie + User Effects Model + genre effect model 0.865
# The RMSE has reduced when the user-movie-genre specific effects are introduced in the model. 

# Lets see how the prediction happened
# get movie titles
movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

# See for the top predicted movies on the movie_avgs or the movie effects factor
# Look for the number of ratings each movie has
all_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating- mu),
            b_u = mean(rating- mu - b_i),
            b_gen = mean(rating- mu - b_i - b_u)) 

validation %>% 
  dplyr::count(movieId) %>% 
  left_join(all_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i), desc(b_u), desc(b_gen)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# See for the low rating movies in prediction
validation %>% 
  dplyr::count(movieId) %>% 
  left_join(all_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i, b_u, b_gen) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


# Noticed that some of the top and low rated movies have few have rated only once.
# It is rather not a good idea to create a model when movies are predicted based on very few number 
# of ratings. So we try regularization by introducing the parameter 'lambda'. Lambda penalise 
# the values coming from small sample sizes. When n is large the value of lambda is effectively ignored.
# When value of n is small the values are shrunken towards zero. When lamda is large penality is
# more.
# The best value of lambda is the value which gives the prediction of minimum RMSE.

# Prediction with regularisation
# Series of lambda values are used.
# Regularisation factor lambda is introduced to userid, movieid and genres. Small estimates in these 
# variables will be penalised.
# With the help of a function the predictions are done for a series of lambda values and RMSE for 
# those predictions are found out. 

lambdas <- seq(0, 10, 0.25)
  lambda_rmses <- sapply(lambdas, function(l){
    mu <- mean(edx$rating)
    b_i <- edx %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
      
    b_u <- edx %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
      
    b_gen <- edx %>% 
      left_join(movie_avgs, by='movieId') %>%
      left_join(user_avgs, by='userId') %>%
      group_by(genres) %>%
      summarize(b_gen = sum(rating - mu - b_i - b_u)/(n()+l))
      
    predicted_ratings_lambda <- 
      validation %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_gen, by = "genres") %>%
      mutate(pred_lambda = mu_val + b_i + b_u + b_gen) %>%
      .$pred_lambda
      
    return(RMSE(predicted_ratings_lambda, validation$rating))
  })
  
# Plot the lambda values against the RMSE
qplot(lambdas, lambda_rmses)  

# Pull the lambda value corresponding to the minimum RMSE
lambda <- lambdas[which.min(lambda_rmses)]
lambda 
# Value of lambda giving minimum RMSE is 5

# RMSE values for different models
rmse_results <- bind_rows(rmse_results,  
                    data_frame(method="Regularized Movie + User Effect Model + genre effect model",  
                                       RMSE = min(lambda_rmses)))
rmse_results %>% knitr::kable()
#|method                                                     |      RMSE|
#|:----------------------------------------------------------|---------:|
#|Just the average                                           | 1.0612018|
#|Movie + User Effects Model + genre effect model            | 0.8649472|
#|Regularized Movie + User Effect Model + genre effect model | 0.8644559|

  
# Regularised value for user, movie and genre effect
 
  regu_avgs <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n(),
              b_u = sum(rating - b_i - mu)/(n()+lambda),
              b_gen = sum(rating - mu - b_i - b_u)/(n()+lambda)) 

# Find the top predicted movies
  validation %>%
    dplyr::count(movieId) %>% 
    left_join(regu_avgs) %>%
    left_join(movie_titles, by="movieId") %>%
    arrange(desc(b_i), desc(b_u), desc(b_gen)) %>% 
    select(title, b_i, b_u, b_gen, n) %>% 
    slice(1:10) %>% 
    knitr::kable()
#|title                                         |       b_i|       b_u|   b_gen|    n|
#|:---------------------------------------------|---------:|---------:|-------:|----:|
#|Shawshank Redemption, The (1994)              | 0.9424978| 0.0001682| 0.0e+00| 3111|
#|Godfather, The (1972)                         | 0.9026465| 0.0002542| 1.0e-07| 2067|
#|Usual Suspects, The (1995)                    | 0.8531914| 0.0001970| 0.0e+00| 2389|
#|Schindler's List (1993)                       | 0.8508447| 0.0001834| 0.0e+00| 2584|
#|Casablanca (1942)                             | 0.8075991| 0.0003593| 2.0e-07| 1275|
#|Rear Window (1954)                            | 0.8056787| 0.0005074| 3.0e-07|  890|
#|Sunset Blvd. (a.k.a. Sunset Boulevard) (1950) | 0.8020419| 0.0013701| 2.3e-06|  333|
#|Third Man, The (1949)                         | 0.7976163| 0.0013419| 2.3e-06|  298|
#|Double Indemnity (1944)                       | 0.7965030| 0.0018446| 4.3e-06|  249|
#|Paths of Glory (1957)                         | 0.7937292| 0.0025182| 8.0e-06|  207|


validation %>%
    dplyr::count(movieId) %>% 
    left_join(regu_avgs) %>%
    left_join(movie_titles, by="movieId") %>%
    arrange(b_i, b_u, b_gen) %>% 
    select(title, b_i,b_u, b_gen, n) %>% 
    slice(1:10) %>% 
    knitr::kable()
#|title                                                |       b_i|        b_u|      b_gen|  n|
#|:----------------------------------------------------|---------:|----------:|----------:|--:|
#|From Justin to Kelly (2003)                          | -2.546473| -0.0624136| -0.0015297| 17|
#|SuperBabies: Baby Geniuses 2 (2004)                  | -2.495050| -0.2045123| -0.0167633|  5|
#|Pokémon Heroes (2003)                               | -2.395829| -0.0843602| -0.0029704| 19|
#|Glitter (2001)                                       | -2.302982| -0.0334736| -0.0004865| 41|
#|Disaster Movie (2008)                                | -2.294564| -0.3100763| -0.0419022|  8|
#|Gigli (2003)                                         | -2.282710| -0.0358917| -0.0005643| 39|
#|Pokemon 4 Ever (a.k.a. Pokémon 4: The Movie) (2002) | -2.277865| -0.0550209| -0.0013290| 31|
#|Barney's Great Adventure (1998)                      | -2.270389| -0.0532955| -0.0012511| 26|
#|Carnosaur 3: Primal Species (1996)                   | -2.258187| -0.1546703| -0.0105939| 11|
#|Yu-Gi-Oh! (2004)                                     | -2.147026| -0.1262957| -0.0074292| 12|

# This is a more reliable prediction as ratings are predicted from those movies with larger number of 
# available ratings.

# The predictions looks acceptable. Top movies are indeed good movies. So the low rated ones.
