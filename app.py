from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
import chess
import chess.svg
import os
from corners_detection import find_corners
from recap import CfgNode as CN

app = Flask(__name__)

# Load YOLO model
model = YOLO(r"yolov8_custom.pt")

# Load corner detection config
cfg = CN.load_yaml_with_base(r'corner_detection.yaml')

def perspective_transform(image, corners):
    """
    Apply a perspective transform to the image based on the given corner points.

    Args:
        image (numpy.ndarray): The input image to be transformed.
        corners (list of tuple): List of corner points in the order [top-right, top-left, bottom-left, bottom-right].

    Returns:
        numpy.ndarray: The transformed image with a top-down view.
    """
    top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]

    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype="float32")

    matrix, mask = cv2.findHomography(corners, dimensions)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    resized_image = cv2.resize(warped, (640, 640))

    return resized_image


def detect_objects_in_image(image, model, output_path="output.jpg"):
    """
    Perform object detection using the YOLO model on the provided image.

    Args:
        image (numpy.ndarray): The input image to detect objects in.
        model (YOLO): The YOLO model used for detection.
        output_path (str): The file path where the output image will be saved.

    Returns:
        tuple: A tuple containing the detection results and the output path.
    """
    results = model(image)
    for result in results:
        annotated_image = result.plot()

    cv2.imwrite(output_path, annotated_image)
    print(f"Results saved to {output_path}")

    return results, output_path


def yolo_to_fen(results, perspective='white'):
    """
    Convert YOLO output to FEN notation, ensuring only the highest-confidence detection is used for each piece.

    Args:
        results (list): List of ultralytics.engine.results.Results objects (one per image).
        perspective (str): Chessboard perspective ('white' or 'black').

    Returns:
        str: FEN string representing the board state.
    """
    board = [['' for _ in range(8)] for _ in range(8)]

    piece_mapping = {
        'black-bishop': 'b',
        'black-king': 'k',
        'black-knight': 'n',
        'black-pawn': 'p',
        'black-queen': 'q',
        'black-rook': 'r',
        'white-bishop': 'B',
        'white-king': 'K',
        'white-knight': 'N',
        'white-pawn': 'P',
        'white-queen': 'Q',
        'white-rook': 'R'
    }

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    square_detections = {}

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        square_x = int(center_x / (640 / 8))
        square_y = int(center_y / (640 / 8))

        if perspective == 'black':
            square_x = 7 - square_x
            square_y = 7 - square_y

        class_name = result.names[int(class_id)]

        if class_name in piece_mapping:
            piece_fen = piece_mapping[class_name]

            if (square_x, square_y) in square_detections:
                if confidence > square_detections[(square_x, square_y)][1]:
                    square_detections[(square_x, square_y)] = (piece_fen, confidence)
            else:
                square_detections[(square_x, square_y)] = (piece_fen, confidence)

    for (square_x, square_y), (piece_fen, _) in square_detections.items():
        board[square_y][square_x] = piece_fen

    fen_rows = []
    for row in board:
        fen_row = ''
        empty_count = 0
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen = '/'.join(fen_rows)
    fen += " w - - 0 1"
    return fen

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    perspective = request.form.get('perspective', 'white')

    # Save the uploaded file
    image_path = "uploaded_image.jpg"
    file.save(image_path)

    # Process the image
    img = cv2.imread(image_path)
    corners = find_corners(cfg, img)
    img_t = perspective_transform(img, corners)
    results, output = detect_objects_in_image(img_t, model)
    fen = yolo_to_fen(results, perspective)
    board = chess.Board(fen)
    svg_output = chess.svg.board(board)

    # Save SVG
    svg_path = "static/chess_board.svg"
    with open(svg_path, "w") as f:
        f.write(svg_output)

    # Return results
    return jsonify({
        "fen": fen,
        "svg_url": f"/{svg_path}",
        "output_image": f"/{output}"
    })
@app.route("/static/chess_board.svg")
def generate_svg():
    fen = request.args.get('fen', '8/8/8/8/8/8/8/8 w - - 0 1')
    try:
        board = chess.Board(fen)
        svg_output = chess.svg.board(board)
        return Response(svg_output, mimetype='image/svg+xml')
    except ValueError:
        return jsonify({"error": "Invalid FEN notation"}), 400
    

if __name__ == "__main__":
    app.run(debug=True)