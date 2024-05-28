import sklearn.ensemble
import torch 
import sklearn 


def compare_samples_classifier_based(P: torch.tensor, 
                                     Q:torch.tensor, 
                                     used_model: sklearn.base.BaseEstimator = sklearn.ensemble.RandomForestClassifier(),
                                     n_folds = 10,
                                     balance_classes = True) -> dict:
    """
    A function that compares two samples from two distributions using a classifier
    Args:
        P: torch.tensor: the samples from the first distribution
        Q: torch.tensor: the samples from the second distribution
        used_model: sklearn.base.BaseEstimator: the classifier to use
        n_folds: int: the number of folds to use in the cross validation
        balance_classes: bool: whether to use as much samples from one class as from the other
    Returns:
        dict: a dictionary containing the result in terms of roc_auc_score and accuracy
    """

    n = P.shape[0]
    m = Q.shape[0]

    if balance_classes:
        n = min(n, m)
        m = n
        P = P[:n]
        Q = Q[:m]

    X = torch.cat([P, Q], dim = 0)
    y = torch.cat([torch.zeros(n), torch.ones(m)]).numpy()

    cv = sklearn.model_selection.StratifiedKFold(n_splits = n_folds)

    roc_auc_scores = []
    accuracy_scores = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        used_model.fit(X_train, y_train)

        y_pred = used_model.predict(X_test)
        roc_auc_scores.append(sklearn.metrics.roc_auc_score(y_test, y_pred))
        accuracy_scores.append(sklearn.metrics.accuracy_score(y_test, y_pred))

    return {"CST": {"accuracy": torch.tensor(accuracy_scores).mean().item(), "roc_auc_score": torch.tensor(roc_auc_scores).mean().item()}}


