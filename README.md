# Crop Feature Selection

This project provides tools for evaluating the predictive power of soil features for crop classification using logistic regression. It includes data loading, model training, and feature evaluation utilities, along with tests to ensure correctness.

## Project Structure

```
data/
    soil_measures.csv         # Dataset containing soil measurements and crop labels
src/
    model.py                  # Core logic for data loading, splitting, and feature evaluation
tests/
    test_model.py             # Unit tests for model functions
```

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sheikh-mohammad-rakib/crop-feature-selection.git
    cd crop-feature-selection
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
   *(Create a `requirements.txt` with packages like `pandas`, `scikit-learn`, and `pytest` if not present.)*

### Usage

#### Data

Place your soil measurement data in `data/soil_measures.csv`. The CSV should contain feature columns and a target column for crop labels.

#### Running Feature Evaluation

You can use the functions in [`src/model.py`](src/model.py) to:
- Load data: `load_data(path)`
- Split data: `train_test_split_data(df)`
- Evaluate features: `evaluate_features(df)`

Example usage:
```python
from src.model import load_data, evaluate_features

df = load_data("data/soil_measures.csv")
results = evaluate_features(df)
print(results)
```

#### Running Tests

To run the tests:
```sh
pytest tests/
```

## File Descriptions

- [`src/model.py`](src/model.py): Implements data loading, train/test splitting, single-feature model training, and feature evaluation.
- [`tests/test_model.py`](tests/test_model.py): Contains tests for data loading and feature evaluation functions.
- [`data/soil_measures.csv`](data/soil_measures.csv): Example dataset (replace with your own data).

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

## License

MIT License
