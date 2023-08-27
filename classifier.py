import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
 
df = pd.read_csv("Datasets/gamespot_game_reviews.csv")

### CREATING A NEW DATAFRAME WITH JUST THESE COLUMNS
reviews_df = df[['tagline', 'classifier']]

### MAKES THE tagline COLUMN A LIST TO ANALYZE IT WITH THE CountVectorizer
tagline_list = reviews_df['tagline'].tolist()
count_vect = CountVectorizer()

### TRANSFORMS THE WORDS INTO NUMBERS AND THEN COUNTS FREQUENCY
x_train_counts = count_vect.fit_transform(tagline_list)

### CALCULATES THE IMPORTANCE OF WORDS BASED ON FREQUENCY
tfidf = TfidfTransformer()
x_train_tfidf = tfidf.fit_transform(x_train_counts)
 
### CREATING OUR TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(
    x_train_tfidf,
    np.array(reviews_df['classifier']),
    test_size=0.3,
    random_state=0)

### APPLY THE MACHINE LEARNING MODEL TO THE SETS
classification_model = MultinomialNB().fit(X_train, y_train)
y_pred = classification_model.predict(X_test)

## CALCULATES THE EFFECTIVENESS OF THE MACHINE LEARNING MODEL
number_right = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        number_right +=1
 
print(f"Accuracy for tagline classify: {number_right/float(len(y_test)) * 100}%")
