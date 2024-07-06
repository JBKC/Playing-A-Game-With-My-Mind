import crown_processing as processing
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# class 1 = raise right arm
# class 2 = raise left arm

def model_run():

    def plot():
        pca = PCA(n_components=2)
        X_train_2d = pca.fit_transform(X_train)

        h = 0.02  # step size in the mesh
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # predict the probability ranges for each class
        Z = lda.predict_proba(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
        Z = Z.reshape(xx.shape)

        contour = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)  # plot probability gradient
        # plt.colorbar(contour, label='Class 1 Probability')

        plt.scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1], c='blue', edgecolor='k',
                    label='Class 0')
        plt.scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1], c='red', edgecolor='k',
                    label='Class 1')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.show()

    P1, P2 = processing.main()

    X = np.concatenate((P1, P2), axis=0)                            # training data
    y = np.concatenate((np.zeros(P1.shape[0]), np.ones(P2.shape[0])))      # labels

    print(X.shape)

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

    plot()


if __name__ == '__main__':
    model_run()