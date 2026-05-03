from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_prediction_model(df, target):
    df = df.select_dtypes(include='number')

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return model, score