```markdown
# Protein Structure Prediction Web App
Demo App Link: https://proteinstructurepredictionusingesmfold.streamlit.app/

This repository contains a web application for predicting protein structures using the ESMFold model from Meta. The app is built with Python and Streamlit, providing an interactive interface for users to input protein sequences and visualize the predicted 3D structures.

## Introduction
Protein structure prediction is a crucial task in biochemistry and molecular biology. This web app leverages the ESMFold model to predict protein structures from amino acid sequences. The app is designed to be user-friendly and educational, making it a valuable tool for students, educators, and researchers.

## Features
- **Interactive Interface**: Built with Streamlit for easy use.
- **Protein Structure Visualization**: Uses py3Dmol and stmol for 3D visualization.
- **ESMFold Integration**: Predicts protein structures using the ESMFold model from Meta.
- **Downloadable Results**: Allows users to download the predicted PDB files.

Installation
To run this app locally, you'll need to install the required libraries. You can do this using pip:


pip install streamlit stmol py3Dmol requests biotite altair pillow
```

## Usage
1. Clone this repository:
    ```bash
    git clone https://github.com/vishalbhoir18/proteinstructurepredictionusingESMFold.git
    cd proteinstructurepredictionusingESMFold
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

3. Open your web browser and go to `http://localhost:8501` to access the app.

4. Enter a protein sequence (up to 400 amino acids) in the sidebar and click the "Predict" button to visualize the predicted structure.

## Acknowledgements
- This app is created with instructions provided by Chanin Nantasenamat [Data Professor](https://github.com/dataprofessor)
- Credit: This app is inspired by [ESMFold on Hugging Face](https://huggingface.co/spaces/osanseviero/esmfold).

---

Happy coding, and may your proteins always fold correctly! 🧬✨
```
