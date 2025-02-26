# Chessboard Detection and FEN Generation

![Chessboard Detection](https://img.shields.io/badge/Chessboard-Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Custom%20Model-yellow)

A web application that detects chess pieces on a chessboard image, applies perspective transformation, and generates FEN (Forsyth-Edwards Notation) notation. The application also provides an interactive interface to edit the FEN and visualize the chessboard.

---

## Features

- **Chessboard Detection**: Detects chess pieces using a custom YOLOv8 model.
- **Perspective Transformation**: Corrects the perspective of the chessboard image.
- **FEN Generation**: Converts detected pieces into FEN notation.
- **Interactive Interface**: Edit the FEN and visualize the updated chessboard.
- **Lichess Integration**: Open the position in Lichess for analysis.
- **Dark Theme**: Sleek and modern dark theme for better user experience.

---
## Dataset and Training

### Chess Pieces Dataset
The model was trained using the **Chess Pieces Dataset** from Roboflow. You can access the dataset here:  
[Chess Pieces Dataset on Roboflow](https://universe.roboflow.com/fhv/chess-pieces-2-6l8qq/dataset/5)

### Training Notebook
The training process was conducted on Kaggle. You can view the training notebook here:  
[Chess Object Detection Using FHV Data on Kaggle](https://www.kaggle.com/code/sum1tk/chess-object-detection-using-fhv-data)

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [API Endpoints](#api-endpoints)
4. [Technologies Used](#technologies-used)
5. [Contributing](#contributing)
6. [License](#license)

---

## Installation

### Prerequisites

- Python 3.8+
- Flask
- OpenCV
- Ultralytics (YOLOv8)
- Chess library

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chessboard-detection.git
   ```
2. Create a virtual environment:

    ```bash
    
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ````
4. Run the Flask application:

    ```bash
    python app.py
    ```
6. Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---
## Usage

### Using the Web Application

1. **Upload an Image**:
   - Drag and drop a chessboard image or click to upload.
   - Select the perspective (White or Black).

2. **Process the Image**:
   - The application will detect the chess pieces, correct the perspective, and generate the FEN notation.

3. **Edit FEN**:
   - You can edit the FEN notation and update the chessboard visualization.

4. **Analyze on Lichess**:
   - Click the link to open the position in Lichess for further analysis.

---

### Using the `image2fen.py` Script

The `image2fen.py` script allows you to process a chessboard image from the command line. It takes two arguments: the image path and the perspective (`white` or `black`). It saves the SVG of the chessboard and provides a Lichess link.

#### Command

  ```bash
  python image2fen.py <image_path> <perspective>
  ```
**Arguments**

`<image_path>`: Path to the chessboard image file.

`<perspective>`: Chessboard perspective (white or black).

**Example**

  ```bash
  
  python image2fen.py chessboard.jpg white
  ```
**Output**

1. **FEN Notation**:
   1. The script will print the FEN notation of the detected chessboard.
1. **SVG File**:
   1. The script will save the chessboard visualization.
1. **Lichess Link**:
   1. The script will print a link to analyze the position on Lichess.

-----
## API Endpoints

**POST /process**

- **Description**: Processes the uploaded chessboard image and returns the FEN notation, SVG, and output image.
- **Request**:
  - file: Chessboard image file.
  - perspective: Chessboard perspective (white or black).
- **Response**:

    ```json
    
    
    
    {
    
    "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
    
    "svg_url": "/static/chess_board.svg",
    
    "output_image": "/output.jpg"
    
    }
  ```
**GET /static/chess\_board.svg**

- **Description**: Generates an SVG of the chessboard based on the provided FEN notation.
- **Query Parameters**:
  - fen: FEN notation (default: 8/8/8/8/8/8/8/8 w - - 0 1).
- **Response**: SVG image.
-----
## Technologies Used

- **Python**: Core programming language.
- **Flask**: Web framework for the backend.
- **YOLOv8**: Object detection model for chess piece detection.
- **OpenCV**: Image processing and perspective transformation.
- **Chess Library**: FEN generation and SVG rendering.
- **HTML/CSS/JavaScript**: Frontend interface.
-----
## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
1. Create a new branch (git checkout -b feature/YourFeature).
1. Commit your changes (git commit -m 'Add some feature').
1. Push to the branch (git push origin feature/YourFeature).
1. Open a pull request.
-----
## License

This project is licensed under the MIT License. See the [LICENSE](https://license/) file for details.

-----
## Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8.
- [Lichess](https://lichess.org/) for providing an analysis platform.
- [Chess Library](https://python-chess.readthedocs.io/) for FEN and SVG handling.
- **Georg Wölflein and Ognjen Arandjelović** for the corner detection script and recap library for config management. 
  
    "Determining Chess Game State from an Image,"  
    *Journal of Imaging*, vol. 7, no. 6, article no. 94, 2021.  
    DOI: [10.3390/jimaging7060094](https://doi.org/10.3390/jimaging7060094).

- **Roboflow** for providing the chess pieces dataset:
  - Dataset: [Chess Pieces Dataset on Roboflow](https://universe.roboflow.com/fhv/chess-pieces-2-6l8qq/dataset/5)

- **Kaggle** for providing the platform to train the model:
  - Training Notebook: [Chess Object Detection Using FHV Data on Kaggle](https://www.kaggle.com/code/sum1tk/chess-object-detection-using-fhv-data)

