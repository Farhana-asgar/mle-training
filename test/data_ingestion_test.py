import os

import pytest

from house_value_prediction import data_ingestion

# Constants for testing
MOCK_HOUSING_URL = "http://mock.url/housing.tgz"
MOCK_HOUSING_PATH = "mock/path/housing"


@pytest.fixture
def mock_urlretrieve(mocker):
    return mocker.patch("urllib.request.urlretrieve", return_value=None)


@pytest.fixture
def mock_tarfile_open(mocker):
    mock_tar = mocker.Mock()
    # Mocking the tarfile.open method correctly
    return mocker.patch("tarfile.open", return_value=mock_tar)


@pytest.fixture
def mock_makedirs(mocker):
    return mocker.patch("os.makedirs", return_value=None)


def test_fetch_housing_data(mock_urlretrieve,
                            mock_tarfile_open, mock_makedirs):

    data_ingestion.fetch_housing_data(housing_url=MOCK_HOUSING_URL,
                                      housing_path=MOCK_HOUSING_PATH)

    # Assertions
    mock_makedirs.assert_called_once_with(MOCK_HOUSING_PATH, exist_ok=True)

    mock_urlretrieve.assert_called_once_with(MOCK_HOUSING_URL, os.path.join(
        MOCK_HOUSING_PATH, "housing.tgz"))

    # Check that tarfile.open was called with the correct path
    mock_tarfile_open.assert_called_once_with(os.path.join(
        MOCK_HOUSING_PATH, "housing.tgz"))

    # Check that extractall was called
    mock_tarfile_open().extractall.assert_called_once_with(
        path=MOCK_HOUSING_PATH)

    # Check that close was called
    mock_tarfile_open().close.assert_called_once()
