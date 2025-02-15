import matplotlib.pyplot as plt
from chesscog.corner_detection.detect_corners import find_corners
import cv2
from recap import URI, CfgNode as CN


def detect_board(image, cfg):
    corners = find_corners(cfg, image)
    return corners

if __name__ == "__main__":
    image_path = "/Users/sahilchaddha/Downloads/WhatsApp Image 2025-02-12 at 03.56.22.jpeg"

    filename = URI(image_path)
    image = cv2.imread(str(filename))

    cfg = CN.load_yaml_with_base("config://corner_detection.yaml")

    corners = detect_board(image, cfg)

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Corner detection output")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(*corners.T, c="r")
    plt.axis("off")
    plt.show()
