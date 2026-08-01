# Dataset

**KDD Cup 2015 — Student Dropout Prediction**

Source: https://www.kaggle.com/datasets/sst2023/kdd-cup-2015

Download the dataset from the link above and place the files in this folder
following the structure below. These files are gitignored (`*.csv`, `data/`)
and are not committed to this repository — download them separately.

## Expected structure
data/
├── object.csv # course structure / module hierarchy
├── date.csv # course start/end dates
├── train/
│ ├── enrollment_train.csv # enrollment_id, username, course_id
│ ├── log_train.csv # clickstream events (largest file, ~618MB)
│ └── truth_train.csv # labels: enrollment_id, dropout (1/0)
└── test/
├── enrollment_test.csv
├── log_test.csv # ~408MB
└── truth_test.csv


## Notes

- `log_train.csv` and `log_test.csv` are too large for pandas to load
  comfortably — these are read via PySpark for all processing.
- `enrollment_*.csv`, `truth_*.csv`, `object.csv`, and `date.csv` are small
  enough to explore directly with pandas.
- `date.csv` gives per-course start/end dates, useful for deriving
  recency/time-remaining features during feature engineering.
