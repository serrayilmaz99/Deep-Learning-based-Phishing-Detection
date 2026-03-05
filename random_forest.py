import pandas as pd
from feature_main import training_df2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay



training_df2 = shuffle(training_df2)
phishing_t = training_df2['phishing'] 
training_df_t = training_df2.drop(columns=['phishing'])


X_train, X_test, y_train, y_test = train_test_split(training_df_t, phishing_t, train_size=0.8, random_state=42)
X_train.shape, X_test.shape

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

classifier_rf = RandomForestClassifier(max_depth=5, min_samples_leaf=5, n_estimators=30,
                       n_jobs=-1, random_state=42,oob_score=True)

classifier_rf.fit(X_train, y_train)
y_pred=classifier_rf.predict(X_test)


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(classifier_rf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
print(feature_imp)

cm = confusion_matrix(y_test, y_pred, labels=classifier_rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier_rf.classes_)
disp.plot(colorbar=False)
plt.show()

print(f1_score(y_test, y_pred, average=None))
print(mean_squared_error(y_test,y_pred))