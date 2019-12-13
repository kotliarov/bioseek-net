import sklearn.model_selection

def make_train_validation_test(X, train_size, validation_size, random_state=12345):
    """
    :X: index array
    :train_size: ratio of train data set
    :validation_size: ratio of validation data set
    :random_state: seed
    """
    x_train, x_other = sklearn.model_selection.train_test_split(X, 
                                                                train_size=train_size, 
                                                                random_state=random_state)
    train_size = validation_size / (1. - train_size)
    x_valid, x_test  = sklearn.model_selection.train_test_split(x_other, 
                                                                train_size=train_size,
                                                                random_state=random_state)
    return x_train, x_valid, x_test
    


