import cv2
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix

f =open('hog.pckl', 'rb')
HOG_data = pickle.load(f)
f.close()

x = HOG_data[:, 1:HOG_data.shape[1]]
y = HOG_data[:, 0]

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_test, x_validate, y_test, y_validate = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x_train, y_train)

y_test_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_test_pred)
tn = cm[1][1]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[0][0]

acc = (tp+tn)/(tn+fp+fn+tp)
print(acc)
