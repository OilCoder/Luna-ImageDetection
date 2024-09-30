# Luna-ImageDetection: Slickline Tool Recognition

This project aims to recognize slickline tools in unlabeled photos using machine learning techniques. It leverages computer vision and deep learning to automate the identification of various slickline tools used in the oil and gas industry.

## Project Structure

- `src/`: Contains the source code for the project
- `data/`: Directory for storing image data (not tracked by Git)
- `models/`: Directory for storing trained models (not tracked by Git)

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/OilCoder/Luna-ImageDetection.git
   cd Luna-ImageDetection
   ```
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To prepare data, train the model, and evaluate:
   ```
   python src/main.py
   ```

2. To run the Streamlit app for predictions:
   ```
   python src/run_app.py
   ```

## Contributing

Contributions to improve Luna-ImageDetection are welcome. Please feel free to submit a Pull Request.

## License

[License information to be added]

## About the Author

This project is maintained by Carlos Esquivel, a specialist in applying machine learning techniques to petrophysical data. For more projects related to oil and gas industry data analysis, visit [github.com/OilCoder](https://github.com/OilCoder).