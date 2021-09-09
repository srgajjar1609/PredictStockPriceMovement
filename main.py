import pandas as pd
df = pd.read_csv('Data.csv',encoding="ISO-8859-1")
print(df.head())


#After readind the file we will do divide the data into taraining and testing data
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# feature Engineering
# we have different : , . etc are in the data so we don't need these for analysis

#removing punctuations
#data = train.iloc[:, 2:27] -------> I have took the 2 to 27 independent features and apply the regular expression....
data = train.iloc[:, 2:27]

#The regex checks for a dash(-) followed by a numeric digit (represented by d) and replace that with an empty string and the inplace
#parameter set as True will update the existing series.
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# renaming column names for ease of access
# replacing all the column names by characters by the numbers
list1 = [i for i in range(25)]
new_Index = [str(i) for i in list1]
data.columns = new_Index
print(data.head(5))


#converting headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower()
print(data.head(1))



#read all the inddex and table values and combine them in one paragraph
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))
print(headlines[0])
#print(headlines)

#Countvectorizer is basicallly does that takes sentances and it convert it into vector of features
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

######## implement BagofWords

#gram_range: An n-gram is just a string of n words in a row. E.g. the sentence ‘I am Groot’ contains the 2-grams ‘I am’ and ‘am Groot’.
# The sentence is itself a 3-gram. Set the parameter ngram_range=(a,b) where a is the minimum and b is the maximum size of ngrams you want to include in your features.
# The default ngram_range is (1,1).The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used.
# For example an ngram_range of (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams. Only applies if analyzer is not callable.
countvector = CountVectorizer(ngram_range=(2,2))

#this will be our dataset
traindataset = countvector.fit_transform(headlines)


#implement RandomForest Classifier
randomclassifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
randomclassifier.fit(traindataset, train['Label'])


##Predict for the Test Dataset
test_transform = []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))
test_dataset = countvector.transform(test_transform)
prediction = randomclassifier.predict(test_dataset)
print('Prediction:::::::::::::::::')
print(prediction)

## Import libraries to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix = confusion_matrix(test['Label'], prediction)
print('Matrix:::::::::::::::::')
print(matrix)

score = accuracy_score(test['Label'], prediction)
print('Score:::::::::::::::::')
print(score)

report = classification_report(test['Label'], prediction)
print('Report:::::::::::::::::')
print(report)