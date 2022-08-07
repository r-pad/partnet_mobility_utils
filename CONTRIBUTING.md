# Contributing

## Install
```
pip install -e ".[develop]"

pre-commit install
```

## Run CI locally
To run the CI locally:

Setup (make sure docker is installed):
```
brew install act
```

Run act
```
act -j develop
```

## Create data lists
```
python scripts/detect_invalid.py ~/datasets/partnet-mobility/dataset ./partnet_mobility_utils/data_lists/
```
