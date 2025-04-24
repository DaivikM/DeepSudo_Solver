import streamlit as st
import numpy as np
import cv2
import csv
from tensorflow.keras.models import load_model  # type: ignore

from main import SudokuSolver


class SudokuApp:
    def __init__(self):
        st.title("🧩 Sudoku Solver")
        self.model_path = "model/best_model.keras"
        self.run()

    def run(self):
        try:
            input_method = st.radio("Choose input method:", ("Upload Image", "Upload CSV", "Manual Input"))
            if input_method == "Upload Image":
                self.handle_image_upload()
            elif input_method == "Upload CSV":
                self.handle_csv_upload()
            elif input_method == "Manual Input":
                self.handle_manual_input()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    def handle_image_upload(self):
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                st.image(image, caption='Uploaded Image', use_container_width=True)

                # Write image temporarily
                image_path = f"temp_{uploaded_file.name}"
                cv2.imwrite(image_path, image)

                solver = SudokuSolver(image_path=image_path, model_path=self.model_path)
                solved_sudoku = solver.solve_sudoku_image()

                st.success("Solved Sudoku:")
                st.write(solved_sudoku)
            except Exception as e:
                st.error(f"Error processing the image: {e}")

    def handle_csv_upload(self):
        solver = SudokuSolver()
        csv_file = st.file_uploader("Upload CSV file", type=["csv"])
        if csv_file is not None:
            try:
                if solver.load_from_csv(csv_file):
                    st.write("Sudoku Puzzle:")
                    st.table(solver.board)
                    if solver.solve_sudoku():
                        st.success("Sudoku Solved:")
                        st.table(solver.board)
                    else:
                        st.error("No solution exists for the provided puzzle.")
            except Exception as e:
                st.error(f"Error processing the CSV: {e}")

    def handle_manual_input(self):
        solver = SudokuSolver()
        st.write("Enter the Sudoku values (0 for empty):")
        manual_input = []

        try:
            for i in range(9):
                row = st.text_input(f"Row {i+1} (comma-separated)", "0,0,0,0,0,0,0,0,0")
                manual_input.append([
                    int(val.strip()) if val.strip().isdigit() else 0
                    for val in row.split(",")
                ])
            if st.button("Solve"):
                solver.board = manual_input
                st.write("Sudoku Puzzle:")
                st.table(solver.board)
                if solver.solve_sudoku():
                    st.success("Sudoku Solved:")
                    st.table(solver.board)
                else:
                    st.error("No solution exists.")
        except Exception as e:
            st.error(f"Invalid input: {e}")


if __name__ == "__main__":
    SudokuApp()
