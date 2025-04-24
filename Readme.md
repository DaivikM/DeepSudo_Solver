
## 🧩 DeepSudo_Solver - Sudoku Solver with Deep Learning & Streamlit

This project combines deep learning and image processing to recognize and solve Sudoku puzzles from images, CSV files, or manual input. The app is powered by a Convolutional Neural Network (CNN) for digit recognition and uses classical backtracking to solve the puzzle. Built with **TensorFlow**, **OpenCV**, and **Streamlit**.

---

## 🚀 Features

- ✅ Image upload for automatic grid extraction and digit recognition
- ✅ CSV input for structured Sudoku puzzles
- ✅ Manual input mode for full control
- ✅ Digit recognition using CNN
- ✅ Intelligent image preprocessing (thresholding, perspective transform)
- ✅ Sudoku solver using backtracking
- ✅ Interactive web interface via Streamlit

---

## 🧠 Model Training (`Model_Training/Model.ipynb`)

A CNN model is trained to recognize handwritten digits (0–9) using data generators with data augmentation for robustness.
- **Note**: Source of the Data is from Roboflow and Kaggle. 

### Key Components:
- `ImageDataGenerator` for data preprocessing and augmentation
- 3 Convolutional layers with increasing filters (16 → 32 → 64)
- Dropout and Dense layers
- Trained using `categorical_crossentropy`
- Callbacks: `EarlyStopping`, `ModelCheckpoint`

### Example:
- Explore `Model_Training/Model.ipynb` and follow the steps to train the model

- This will train the model and save it as `Model_Training/model/best_model.keras`.

- Replace this model with the `Model/best_model.keras`.

---

## 🧮 Sudoku Solver Logic (`main.py`)

The `SudokuSolver` class contains:
- Board validation and solving
- Image preprocessing (grayscale, blur, adaptive threshold)
- Contour detection and perspective transform
- Cell extraction and digit prediction using the trained CNN
- Solver using classical backtracking algorithm

---

## 🛠️ Installation 🌐 Web App (`app.py`)

### 1. Clone the Repo
```bash
git clone https://github.com/DaivikM/DeepSudo_Solver.git
cd DeepSudo_Solver
```

### 2. Create Virtual Environment
```bash
conda create -p venv python==3.12.9 -y
conda activate venv/
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. How to Run:
```bash
streamlit run app.py
```
---

### Test_samples:
Some samples of images and csv files are also been attached in the repo for reference.

### Interface Options:
- **Upload Image**: Upload a Sudoku image for digit recognition and solving.
- **Upload CSV**: Upload a `.csv` file with comma-separated Sudoku values.
- **Manual Input**: Enter values row by row in the interface.

### Output:
- Solved Sudoku board (displayed in a table)
- Errors displayed if the image is unclear or puzzle is unsolvable

---

## 📜 License

This project is licensed under the **MIT License**.

You can freely use, modify, and distribute the software with attribution, and without any warranty. See the [LICENSE](LICENSE) file for more details.

---

## 📞 Contact

- 📧 Email: [dmohandm11@gmail.com](mailto:dmohandm11@gmail.com)  
- 💻 GitHub: [DaivikM](https://github.com/DaivikM)

---

## 📚 References

- [Streamlit](https://streamlit.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)

---

### 🚀 Happy Coding! ✨

🧪 Clone this repository to explore an AI-powered Sudoku Solver using deep learning for digit recognition, OpenCV for image processing, and an interactive Streamlit web app for solving puzzles from images, CSV files, or manual input.

---
