from app.model_loader import load_model_bundle, load_model_metadata


def test_model_bundle_loads():
    bundle = load_model_bundle()

    assert "model" in bundle
    assert "preprocessing_objects" in bundle
    assert "processed_feature_names" in bundle
    assert "target" in bundle


def test_model_metadata_loads():
    metadata = load_model_metadata()

    assert metadata["model_name"] == "Linear Regression"
    assert metadata["model_version"] == "v3"
    assert metadata["model_type"] == "LinearRegression"
    assert metadata["test_rmse"] > 0
