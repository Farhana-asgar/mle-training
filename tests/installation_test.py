import house_value_prediction
import house_value_prediction.ingest_data
import house_value_prediction.score
import house_value_prediction.train


def test_import():
    """Test that the package imports correctly."""
    assert house_value_prediction.ingest_data is not None
    assert house_value_prediction.train is not None
    assert house_value_prediction.score is not None

