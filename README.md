
semantic_segmentation/
│── models/              # Model definitions
│   ├── unet.py
│   ├── fcn.py
│   ├── __init__.py
│
│── datasets/            # Data handling
│   ├── cityscapes.py
│   ├── pascal_voc.py
│   ├── transforms.py
│   ├── __init__.py
│
│── training/            # Training logic
│   ├── trainer.py
│   ├── evaluator.py
│   ├── __init__.py
│
│── configs/             # Config files for hyperparameters
│   ├── default.yaml
│   ├── custom.yaml
│
│── utils/               # Utility functions
│   ├── metrics.py
│   ├── visualization.py
│   ├── __init__.py
│
│── main.py              # Main script (entry point)
│── requirements.txt     # Dependencies
│── README.md            # Documentation

