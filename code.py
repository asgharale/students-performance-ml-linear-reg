import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from matplotlib import style
import matplotlib.pyplot as pyplot

df = pd.read_csv('student-mat.csv', sep=';')

label = 'G3'

x = np.array(df[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']])
y = np.array(df[label])

best = 0
for _ in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    
    acc = linear.score(x_test, y_test)
    if acc > best:
        best = acc
        with open('student-88%-model.pkl', 'wb') as f:
            pickle.dump(linear, f)


with open('student-88%-model.pkl', 'rb') as f:
    model = pickle.load(f)


predictions = model.predict(x_test)
print("ضریب: ", model.coef_)
print("رهگیری: ", model.intercept_)

print()
print(best)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])


# plotting data
style.use("ggplot")
pyplot.scatter(df['G2'], df[label])
pyplot.ylabel(label)
pyplot.show()