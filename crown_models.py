import crown_processing as processing
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report

# class 1 = raise right arm
# class 2 = raise left arm

def model_run():
    P1, P2 = processing.main()

    X = np.concatenate((P1, P2), axis=0)                            # training data
    y = np.concatenate((np.zeros(P1.shape[0]), np.ones(P2.shape[0])))      # labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train LDA classifier
    lda = LDA()
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)

    # model outputs
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    model_run()