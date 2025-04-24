import streamlit as st
import numpy as np
import cv2
import csv
from tensorflow.keras.models import load_model  # type: ignore

class SudokuSolver:
    def __init__(self, board=None, image_path=None, model_path=None):
        self.size = 9
        if board is not None:
            self.board = board
        else:
            self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.image_path = image_path
        self.model_path = model_path
        self.sudoku_array = None
        self.model = None

    def is_valid(self):
        for i in range(self.size):
            for j in range(self.size):
                val = self.board[i][j]
                if val < 0 or val > 9 or val != int(val):
                    return False
        return True

    def print_board(self):
        for i in range(9):
            for j in range(9):
                print(self.board[i][j], end=" ")
                if (j + 1) % 3 == 0 and j < 8:
                    print("|", end=" ")
            print()
            if (i + 1) % 3 == 0 and i < 8:
                print("-" * 21)

    def solve_sudoku(self):
        if not self.is_valid():
            return False
        empty_cell = self.find_empty_cell()
        if not empty_cell:
            return True
        row, col = empty_cell
        for num in range(1, 10):
            if self.is_valid_move(row, col, num):
                self.board[row][col] = num
                if self.solve_sudoku():
                    return True
                self.board[row][col] = 0
        return False

    def find_empty_cell(self):
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    return (i, j)
        return None

    def is_valid_move(self, row, col, num):
        for i in range(9):
            if self.board[row][i] == num or self.board[i][col] == num:
                return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.board[start_row + i][start_col + j] == num:
                    return False
        return True

    def load_from_csv(self, file):
        try:
            file.seek(0)
            reader = csv.reader(file.read().decode('utf-8').splitlines())
            self.board = []
            for row in reader:
                self.board.append([int(cell.strip()) if cell.strip() else 0 for cell in row])
            return True
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return False
        

    def preprocess_image(self, image):
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        height, width, _ = image.shape
        # Crop the image
        image = image[5:height-5, 5:width-5]

        # Resize the image to the target size (40x40)
        image = cv2.resize(image, (40, 40))

        # Convert the image to float32
        image = image.astype('float32')

        # Normalize the pixel values to the range [0, 1]
        image /= 255.0

        # Add batch dimension
        image = image[np.newaxis, ...]

        return image

    def solve(self, quiz):
        val = self.next_box(quiz)
        if val is False:
            return True
        else:
            row, col = val
            for n in range(1,10): #n is the possible solution
                if self.possible(quiz,row, col, n):
                    quiz[row][col]=n
                    if self.solve(quiz):
                        return True 
                    else:
                        quiz[row][col]=0
            return 
        
    def next_box(self, quiz):
        for row in range(9):
            for col in range(9):
                if quiz[row][col] == 0:
                    return (row, col)
        return False

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        blur = cv2.GaussianBlur(gray, (3,3),6) 
            #blur = cv2.bilateralFilter(gray,9,75,75)
        threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
        return threshold_img

    def load_model(self):
        self.model = load_model(self.model_path)

    def main_outline(self, contour):
        biggest = np.array([])
        max_area = 0
        for i in contour:
            area = cv2.contourArea(i)
            if area >50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i , 0.02* peri, True)
                if area > max_area and len(approx) ==4:
                    biggest = approx
                    max_area = area
        return biggest ,max_area
    
    def reframe(self, points):
        points = points.reshape((4, 2))
        points_new = np.zeros((4,1,2),dtype = np.int32)
        add = points.sum(1)
        points_new[0] = points[np.argmin(add)]
        points_new[3] = points[np.argmax(add)]
        diff = np.diff(points, axis =1)
        points_new[1] = points[np.argmin(diff)]
        points_new[2] = points[np.argmax(diff)]
        return points_new

    def splitcells(self, img):
        rows = np.vsplit(img,9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r,9)
            for box in cols:
                boxes.append(box)
        return boxes

    def possible(self, quiz, row, col, n):
        #global quiz
        for i in range (0,9):
            if quiz[row][i] == n and row != i:
                return False
        for i in range (0,9):
            if quiz[i][col] == n and col != i:
                return False
        row0 = (row)//3
        col0 = (col)//3
        for i in range(row0*3, row0*3 + 3):
            for j in range(col0*3, col0*3 + 3):
                if quiz[i][j]==n and (i,j) != (row, col):
                    return False
        return True

    def solve_sudoku_image(self):
        sudoku_a = cv2.imread(self.image_path)
        sudoku_a = cv2.resize(sudoku_a, (450, 450))
        threshold = self.preprocess(sudoku_a)
        contour_1 = sudoku_a.copy()
        contour_2 = sudoku_a.copy()
        contour, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_1, contour, -1, (0, 255, 0), 3)
        black_img = np.zeros((450, 450, 3), np.uint8)
        biggest, maxArea = self.main_outline(contour)

        if biggest.size == 0:
            raise ValueError("Sudoku grid could not be detected. Please upload a clearer image.")

        biggest = self.reframe(biggest)
        cv2.drawContours(contour_2, biggest, -1, (0, 255, 0), 10)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(sudoku_a, matrix, (450, 450))

        sudoku_cell = self.splitcells(imagewrap)
        self.load_model()
        output = []
        for cell_array in sudoku_cell:
            preprocessed_image = self.preprocess_image(cell_array)
            prediction = self.model.predict(preprocessed_image)
            probabilities = np.array(prediction)
            predicted_class_index = np.argmax(probabilities)
            output.append(predicted_class_index)
        self.sudoku_array = np.array(output).reshape((9, 9))
        self.solve(self.sudoku_array)
        return self.sudoku_array

