# Manufacturing Equipment Output Prediction

This repository contains a Streamlit web application for predicting manufacturing equipment output based on various machine parameters. The application utilizes a machine learning model to provide output predictions, making it useful for manufacturing process optimization and scenario analysis.

## Repository Structure

```
├── app.py                              # Main Streamlit application
├── manufacturing_dataset_1000_samples.csv  # Sample dataset for training
├── linear_regression_model.pkl         # Saved ML model (created on first run)
├── Dockerfile                          # Docker configuration
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- [Docker](https://www.docker.com/) (optional, for containerized deployment)

---

### 1. Running the Application Locally

**A. Clone the repository:**
```bash
git clone https://github.com/zaidaanshiraz/manufacturing_dataset.git
cd manufacturing_dataset
```

**B. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**C. Run the Streamlit application:**
Make sure `app.py` and `manufacturing_dataset_1000_samples.csv` are in the same directory.
```bash
streamlit run app.py
```
The application will start and can be accessed at [http://localhost:8501](http://localhost:8501).

---

### 2. Building and Running with Docker

**A. Build the Docker image:**
```bash
docker build -t manufacturing-prediction-app .
```

**B. Run the Docker container:**
```bash
docker run -p 8501:8501 manufacturing-prediction-app
```

Now, open your browser and navigate to [http://localhost:8501](http://localhost:8501) to use the application inside Docker.

---

## Notes

- The machine learning model (`linear_regression_model.pkl`) will be created automatically the first time you run the app if it does not exist.
- The sample dataset provided (`manufacturing_dataset_1000_samples.csv`) is used to train the model.

## License

This project is provided under the [MIT License](LICENSE).

## Contact

For questions or suggestions, feel free to open an issue or contact [@zaidaanshiraz](https://github.com/zaidaanshiraz).
