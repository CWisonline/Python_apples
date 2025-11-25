# Apple Tray Image Analysis (Python)

This project processes orchard tray photographs to:

- Detect individual apples using Hough Circle Transform
- Measure apple diameter (in pixels and mm)
- Measure redness from the red channel
- Produce CSV files and overlay PNGs per image

## Folder structure

python_apples/
│
├── data/
│ ├── raw/ - original images
│ └── processed/ - output CSVs & overlays
├── src/
│ ├── apples.py - main analysis script
│ └── utils.py
├── requirements.txt
└── README.md


## Installation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt


## Running

SRC\Apple_counter_(v0.1).py