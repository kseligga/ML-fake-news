# Machine learning project - fake news detection step-by-step
## About project
This project is an exploration of Natural Language Processing (NLP) and supervised machine learning techniques applied to the challenge of fake news detection. It involves the development and evaluation of various classifiers, feature engineering, and ensemble methods to maximize precision in identifying fake news.\
Done in collaboration with [**@CCzarek**](https://github.com/CCzarek/)

### Data source:
https://www.kaggle.com/datasets/mohammadaflahkhan/fake-news-dataset-combined-different-sources?resource=download

Our data has 69045 entries with 4 columns: id, title, text and Ground Label (our target variable). In this dataset, approximately 60% of the records are marked as fake news.


## Preprocessing

1. We dropped id column, which does not provide any useful information
2. Instead of dropping duplicated rows, we added a new column with number of occurences - this number might provide us an useful information while classifying news - maybe e.g. fake news spread better and will appear more times in our dataset? 🤔
3. All non-English data, which was consistently marked as fake news, was removed. (around 1,3% of records)
4. **Split data** to training set (70%) and testing set (30%)
5. Handling NA values - we changed them to an empty string, so later they won't ruin our models built based on word vectorization
6. We removed contractions (e.g. changing u -> you, don't -> do not). In Natural Language Processing, contractions are often removed to standardize text and reduce ambiguity. This process, known as “**_decontraction_**”, simplifies the language model’s task by converting multiple possible representations of a phrase into a single, consistent form.
7. We removed _**stop words**_, which are commonly used words such as ‘is’, ‘an’, ‘the’, etc. This was done to focus on important words and reduce the dimensionality of the data. This process helps to emphasize meaningful words and improve the efficiency of algorithms by reducing computational complexity.
8. **_Stemming_** - reducing inflected or derived words to their base or root form. This process helps in reducing the complexity of the model by treating different forms of the same word as a single entity, thereby improving the efficiency and accuracy of text processing tasks.
9. We created **new columns**, describing text structure - counting words in total, proper nouns, commas, exclamation marks and question marks
10. **_Vectorization_** - final, most important step of preprocessing in NLP. It is performed to convert text data into a numerical format that can be processed by machine learning algorithms. \
To avoid overfitting and to speed up the model creation process, we set a minimum and maximum frequency threshold, according to which we will take only a portion of the words into the model.\
After testing model scores with different thresholds, we chose max_df = 0.6 and min_df = 0.01, as other values resulted in lower model scores on the test dataset.


## Building and testing models

To begin, we tried out basic machine learning models, from Python library scikit-learn: 
- ExtraTreeClassifier(),
- RandomForestClassifier(),
- LogisticRegression(),
- GradientBoostingClassifier(),
- DecisionTreeClassifier(),
- XGBClassifier(),
- MultinomialNB()

To somehow measure, how good those models predict target variable in our test set, we use various metrics, such as:
- precision
- accuracy
- recall
- f1
- ROC AUC score

### ROC curve for tested models looks like this:
![](https://i.imgur.com/RsurQjD.png)\
As we see, the best classifiers seems to be XGB, RFT and GBC. Other metrics say more or less the same:

![](https://i.imgur.com/oFWkbHX.png)\


## Analysing our model
Constructing a good model does not merely mean building a model that performs well on the test set. It is crucial to evaluate how our constructed model operates and how it will behave with new data. Therefore, we conduct a feature importance analysis to determine which variables have the most significant impact on the classifier’s decision. For example, here are most important feautures for 2 best classifiers:

XGBoost             |  GBC
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/UdPQtUt.png" alt="image" width="100%" height="auto">  |  <img src="https://i.imgur.com/5vyC0Ni.png" alt="image" width="100%" height="auto">

We clearly see: feature importance for XGBoost model seems balanced, with the columns we artificially added in preprocessing on top. At the same time, GB Classifier bases its prediction heavily on only one variable: the word 'reuters'. Same observation could be seen in other models, in a bit smaller scale. \
\
let's evaluate the performance of our models if we completely remove the word "reuters" from the training and test sets:

![](https://imgur.com/hZYPwP5.png)\
As we see, resulting models are still decent in quality. The best model by every metric, based on XGBC, scored 95,66% in precision and 95,80% in accuracy. Same classifiers performed the best again, but now are they are not based so heavily on only one variable. In this newly created models, GB Classifiers feature importance graph looks much more balanced:

![](https://i.imgur.com/rwXDvy2.png)\


## Pushing the limits
To summarize, the best model we have now is undoubtedly the one based on XGBClassifier. Including the word 'reuters' in model building, it scores 0,996 in ROC AUC and over 96% in other metrics. For models without this word, all metrics still hold at least 95% level, and ROC AUC scoring 0,993. But can we make a model even better than this?\
\
_**Voting**_ in machine learning is an ensemble technique where multiple models' predictions are combined, either by majority vote (_**hard voting**_) or by averaging predicted probabilities (_**soft voting**_), to make a final prediction. This approach can improve model performance by compensating for individual model weaknesses. In hard voting, the class that receives the most votes from the models is selected as the final prediction, while in soft voting, the final prediction is based on the average of the predicted probabilities from the models.\
\
To find the best possible model using voting, we have checked each combination of our previously top4 classifiers (XGBC, GBC, RFC, LR) with either soft or hard voting (hard voting when 2 or more models were included)

![](https://imgur.com/O7lsy20.png)\
The above table compares all of the tested models and their scores on test dataset.\
Using voting, we managed to raise the precision of our models - model that used **hard voting with all of top4 classifiers** performed the best in this metric. In comparison with XGBC alone, this new model has raised precision level by 0,009 (with 'reuters') and 0,005 (without 'reuters'). For the models built without the word 'reuters', also soft voting with RFC and XGBC combined managed to outperform just XGBC in accuracy and f1 metric.
### Conclusion
So, in final conclusion, which one of these models is the best one? As always, the answer is 'it depends'.\
For this specific dataset, it makes no sense to exclude word 'reuters' from the process of making a model, just because it has a great importance in classifying. But if we want to apply our model to new and external data, we need to make it more stable, and not base our decision on containing just one specific word, which is used to mark source of the news.\
We are dealing with a similar problem when it comes to choosing the most important metric - it depends what is going to be a purpose of our model. Precision, which gets to a very decent level of over 96% is the most basic metric, which we should look at if we want to have the biggest percentage of correct predictions. But in life, it often comes down to a decision - is it less worse to have a false positive or a false negative results? For fake news detection, in my opinion, it would be overly meticulous to mark true information as fake news, therefore, precision should be the metric we aim to maximize.
