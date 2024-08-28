# NIKE SHOE PRICE PREDICTION
Members:
Melahat Tayli, Arjun Singh, Artemio Mendoza-Garcia, Paulo Silva, Wa’il Choudar
## Introduction and motivation
Online marketplaces' growth has permanently changed how people buy and sell products,
making it crucial for sellers to price their products on e-commerce platforms accurately.
Unfortunately, over-valued listings are often met with strong resistance from consumers, while
under-valued items may sell slowly due to concerns about authenticity or quality. Predicting
product prices at scale is a complex and challenging task, but it is essential for sellers who want
to maximize their profits and minimize their losses. By studying the factors influencing product
prices, we can gain valuable insights into consumer behavior and preferences, identify potential
pricing strategies and tactics, and uncover opportunities for innovation and improvement on
e-commerce platforms.
## Problem Statement
Can we build a model to predict the cost of a pair of shoes using descriptive
characteristics such as product description or other indicators like color, brand, and customer
rating? We will attempt to study available data in depth to predict the Nike shoe price and draw
conclusions about the predictive power of various indicators.
## Data acquisition and exploratory data analysis
### Scraping Data from Nike.com
We initially reviewed the Kaggle datasets suggested in the list of proposed projects, however,
we found them with little information, sparse in Nike models and features. Therefore we opted
for scraping our data directly from https://www.nike.com/w/shoes-y7ok.
Our first scrape approach leveraged Beautiful Soup with
Selenium web driver for Lazy loading page retrieval. A total of 1,748 rows were extracted, with
31 duplicated; repeated and missing data were deleted. We identified seven potential price
predictors, including shoe model, style, gender and age, marketing description, buyer ratings,
and the number of reviews.
To increase the power of our model, we decided to take a more aggressive approach, and we
impersonated the web search engine programmatically, making use of the same API the
website uses to call the store’s database. The API retrieves other available Nike products, so
the first data cleaning step was filtering products that were not in the category
“FOOTWEAR.”
After a considerable effort reverse-engineering the protocol to make use of the API to traverse
the website, we downloaded approximately 5,000 shoes with a rich set of shoe information: list
price, discount price, name, description, category, color(s), and a series of indicators like if the
shoe was discounted, only for exclusive members, just arrived, coming soon, in stock, sales
channel, and, ratings. We also improved the quality of our dataset: only 64 rows had missing
data, nine fields out of 43 had the same value across the entire dataset, and the overall
completeness was around 98%.
Table 1.1 - a glance at data retrieved from nike.com
##### Removing Unnecessary Columns
After the initial data cleaning mentioned above, we noticed many columns left over are irrelevant
or contain only one unique value (no information gain). These columns contain URL links or
product categories for non-shoe items. We drop these columns from our data set.
##### Feature Engineering
After basic data cleansing (dropping empty rows and rows with only one value or URL
information - product, images, etc.), we encoded the data coming in the field subtitle to expand it
as a categorical variable that we called a subcategory; this subset, along with category, grouped
similar features, which suggested they were hierarchical (hence, the name chosen,
subcategory). Below we show how the prices are distributed among these two groups of
classes.
These subcategories are woman, man, kid, big_kid, boy, little_kids, baby_toddler, infant_toddler,
unisex, toddler, and baby. We turned the subcategory column into dummy variables in our
further preprocessing steps.
#### Categorical Encoding and Tokenization
Several categorical columns describe the properties of the shoe, such as lifestyle, intended
market, and distribution channels. We used sci-kit-learn’s OneHotEncoder to
encode categorical predictors as a one-hot numeric array.
Similarly, in our dataset, there are text-based columns. These are the title, subtitle,
short_description, color-Description, category, and TopColor columns. These text-based
columns were preprocessed with simple_preprocess and preprocess_string functions from the
gensim library. Then, we leverage sci-kit-learn’s CountVectorizer to count tokens and divide the
token counts into their columns (bag of words).
Response Variable
The dataset contains several different price features such as color-FullPrice,
color-CurrrentPrice, fullPrice, currentPrice, etc. This is because the dataset contains what we
call “shoe-level” information (data that describes that shoe type) and “color-level”
information (data that describes that shoe product in a specific color).
Factors such as discount and availability depend on the color, not just the shoe type.
We are interested in predicting the following values:
● Full Price (numerical, price without discount)
● Current Price (numerical, discounted price)
● Sale (boolean indicator, whether the shoe is on sale or not)
In our data exploration steps, we investigated the effect of different features on price values. In
particular, we thought that ratings would be a good predictor for price or discount, but we found
that, at least visually, that was not the case. If any, there was a weak relationship, as indicated in
the graphs below
For modeling purposes, we mainly use fullPrice as our response variable.
Train / Test Split
We perform an 80/20 Train/Test split. Similarly, we perform an 80/20 sub/Val split of the Train.
We use sub/Val to help isolate which models are best without judging against the Test data to
circumvent model selection bias.
MODELING
Baseline Model - Random Forest
Before deciding to go with Random Forest for baseline, we did several exploratory modeling,
including linear regression, polynomial, and logistic regression (to classify discount), and even
we used the stepwise process to try to find how to approach this problem. Most initial
brainstorming models and procedures are organized in a notebook labeled as an appendix to
distinguish from the Main Notebook with the final solution. these procedures are,
In the end, we decided to go for Random Forest. We wanted to take advantage of the RF
capabilities to deal with despair-type data without any previous assumptions. Also, given the
high dimensionality of the design matrix, we thought we could push a large number of ensemble
trees at a relatively depth without having a big overfitting problem.
To establish a baseline for our predictive models, we built a simple RF regression to predict the
Full Price of a Shoe, using the full feature matrix with tokenized predictors to train our data
(feature matrix with 4,557 columns). We got these results
● MSE Train: 20.6484
● MSE Validation: 128.2621
● MAE Train: 2.0923
● MAE Validation: 5.0074
Improving our baseline: hyperparameter Tuning and Feature reduction
Once we got our baseline, the next step was to improve our initial score. For that, we decided to
explore two main paths:
Reducing the number of features. For this task, we chose Lasso Regression because it
tended to decrease the regression coefficient, many of them to zero.
Tuning Random Forest hyperparameters (max_depth and n_models). We performed
cross-validation and looped over different n_models to find the optimal values.
Feature Selections using Lasso
We fitted a Lasso regression in our training data, followed by cross-validation with 5 folds and
an alpha range from 0.01 to 0.8. Using Lasso with cross-validation, we ensure that our feature
selection process is robust and reliable. We aimed to find the best features that minimize
minimum absolute error (MAE). The plots below depict the output for one run. Across different
executions of the procedure, we consistently reduced the number of features from 4,557 to a
number between 600 and 300 without losing prediction power.
The table below depicts the top 10 important features according to the Lasso feature selection
method. Looking at them, we realize that these are vectorized features originally from a
descriptive (text) predictor. This is an important fact to highlight because it supports our original
question to predict the price based on shoe descriptors. To the left, we listed the origin column
from where the vectorized version was deducted:
● ‘Sustainable'
● 'TopColor'
● 'category,
● 'channel'
● 'color-Description'
● 'color-MemberExclusive'
● 'color-sold'
● 'short_description'
● 'subcategory’
● 'subtitle'
● 'title'
##### Tuning max_depth with a Single Decision tree
The next step in our path to improvement is to tune max_depth using a single decision tree and
cross-validation. By optimizing the max depth, we can ensure that our model is not overfitting or
underfitting the data. To tune max_depth, we trained multiple decision trees with different
values for the max depth parameter and evaluated their performance using cross-validation.
The following plot shows the results for a specific run. In this case, our best_depth is 88.
We searched the best max_depth between 1 and 100 on a single decision tree by i.
The last step to improve the baseline is finding the best value for n_estimators.
We searched for the best number of estimators by iterating over a Random Forest model between
100 to 800 with 100 increments, as shown below
For this run, the minimal MAE was achieved for n_estimator = 200,
Comparing Baseline with Tuned Models - Success!
Once we reduce our features and get the hyperparameters tuned, we can test against our
baseline. A back-to-back comparison shows MAE improvement from 5.10 (baseline) to 3.98
(tunned).
Random forest with the best hyperparameters (max depth=88, number of estimators=200). We
used the most important 509 features as our predictors. These features were chosen by the
Lasso regression method. Below we can observe the output from the main project Jupiter
notebook.
##### Altogether - Stacking Models on TEST
After having our models tuned, we decided to go one step further and explore using stacked
models to improve our predictions. Specifically, we used stacked models to predict four different
variables:
The full price of a shoe
The full price for a particular color of a shoe
Whether a color of a shoe model is on sale (classifying sale vs. non-sale)
The discount amount for a particular color of a shoe
We used different models combinations and evaluated the improvements in predictive power.
Prediction Performance of full price of a shoe
First, we use a single Random Forest Tree to predict results over test. Our MAE is 4.69 with
MSE 200.223,
These are not impressive results, but considering it is a single model performed over the Test
dataset, it is not a bad result. We could argue that the result of MAE for the Test dataset
suggests that our model generalizes well.
Prediction Performance for full price at Color Level
We predicted the full price for a specific color, first with a single tree, then stacking of two, and
finally adding more trees up to 10. We can see in the tables below that after the first iteration,
MAE needs to improve.
However, from 1 tree to stacking a second one, there is an improvement in MAE of about $1,
and another 50 cents in adding a third one,
Predict Discount at the Color Level
Then, we switched the type of predictor and went with a classifier to try to predict if a shoe is
discounted or not. For this model, we trained and stacked three trees.
The accuracy obtained is okay but could be more impressive, 70% over the test dataset.
##### Predicted discounted Price for a Particular color of a shoe
Predicting discounted price has a worst error than predicting the full price, regardless if it is at a
shoe level or for a specific color. Even stacking more than one tree, we improve the results as
shown in table 5.6 and 5.7.
M
Extra Grade: Gradient Boosting vs. Random Forest
As a final learning goal, we decided to do a couple of experiments. The first one was to test if a
boosting algorithm could perform better than a Random Forest. For this purpose, we trained a
single Gradient Boosting Regressor using the reduced features and tuned hyperparameters we
found before. The results were that Gradient Boost overperformed Random Forest: Validation
MAE: 2.52, MSE: 83.76, R
2=0.96. This model deserves its own project, with hyperparameter
tuning and stacking. This could be follow-up work.
##### Alternative Feature Selection using Correlation. The second experiment was to use a simple
feature selection strategy and then compared it against Lasso. For that, we used the 300 most
price-correlated features. Then we used that subset to train a Gradient Boosting Regressor and
compare it against our previous GBR baseline model. The results were not better than Lasso,
so we failed to reject the null hypothesis that Lasso is the best predictor. MAE: 6.5, MSE: 127.3
## CONCLUSIONS
In this project, we tested different tree models for predicting the full price, discounted price, and
amount discounted for a shoe and a shoe-specific color. Overall, we got excellent results getting
price predictions with MAE between 3 and 4 dollars difference from the real price, discounted or full.
However, for the amount discounted, the power was lower, getting MAE around 10 and 11 dollars
difference from the actual amount. Also, we fitted a decision tree to predict if a shoe model is on sale
or not, getting an accuracy of 70%. All these results were obtained using Test data that was
untouched until the last step.
Our results showed that using a tuned single Random Forest Tree model provided good prediction
performance for the full price of the shoe but slightly worse performance when predicting the full
price at the color level. Stacking multiple models improved the prediction performance slightly, but
less than expected.
Predicting the discounted price had a worse error than predicting the full price.
As a final step, we fit a single, tuned, single Gradient Boosting model, which resulted in a better
performance than any of the previous Random Forest models, even when stacked. We got an MAE
equal to $2.52 divergence from the actual price. We did not explore stacking boosting models, as
this would’ve expanded the scope of the project beyond the limited time available
Overall, our results suggest that the prediction performance could be improved by using different
models and potentially other features or preprocessing methods. A future project could focus on
stacking classification models, including Gradient Boosting, on improving prediction power.
To conclude this project, we can go back to our original question if we can build a model to predict
the cost of a pair of shoes using descriptive characteristics. As per the experiments
conducted and the results obtained, the evidence suggests that yes, we can.
## REFERENCES
Note: Since the references were submitted in Milestone 2, to save space, we will list them here
without a summary.
1. Mercari Price Suggestion Challenge
https://www.kaggle.com/c/mercari-price-suggestion-challenge/overview
2. Introduction to Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency
https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/
3. Vurvey Releases New Footwear Industry Data, Shows 99% of Consumers Value Inclusivity
https://vurvey.co/resources/customer-experience/consumers-buy-from-inclusive-brands/
4. Kaggle: Adidas VS Nike https://www.kaggle.com/datasets/kaushiksuresh147/adidas-vs-nike
5. Kaggle: StockX Sneaker Data:
https://www.kaggle.com/datasets/hudsonstuck/stockx-data-contest