from src.models.train import train_churn_model

def test_train_churn_model():
    model = train_churn_model()

    assert model.n_features_in_ == 19
    assert list(model.classes_) == [0, 1]
