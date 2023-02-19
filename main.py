import cv2 # pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage.morphology import skeletonize, thin, erosion, dilation, opening, closing, white_tophat
import sknw
import skimage.graph
from tkinter import *
from tkinter.filedialog import askopenfilename

def shortest_path(start, end, binary, pad_width=10):
    costs=np.where(binary,1,1000)
    path, cost = skimage.graph.route_through_array(
        costs, start=start, end=end, fully_connected=False)
    return path,cost

# We don't want a full GUI, so keep the root window from appearing
Tk().withdraw()

# Show an "Open" dialog box and return the path to the selected file
filename = askopenfilename()

# Load the image
img = cv2.imread(filename)

# Convert the image to grayscale
kernel = np.ones((1, 1), np.uint8)

# Convert to GrayScaledImage
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary threshold, Otsu threshold, and binary threshold again to obtain a binary image
# https://muthu.co/otsus-method-for-image-thresholding-explained-and-implemented/#:~:text=Otsu's%20method%5B1%5D%20is%20a,of%20background%20and%20foreground%20pixels.
pog, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
pog, threshold2 = cv2.threshold(threshold, 10, 255, cv2.THRESH_BINARY_INV)
threshold2 = np.where(threshold2 == 255, 1, 0).astype(np.uint8)

# Connect any unconnected white pixels in the binary image using morphological operations
structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
threshold2 = cv2.morphologyEx(threshold2, cv2.MORPH_CLOSE, structuring_element)

# Skeletonize the thresholded image to obtain a skeleton
skel = skeletonize(threshold2)

# Build a network graph from the skeleton
graph = sknw.build_sknw(skel, multi=False)
G = nx.Graph(graph)

# Display the original image
plt.imshow(img, cmap='gray')

# Draw edges on the image
for (s, e) in graph.edges():
    ps = graph[s][e]['pts']
    plt.plot(ps[:, 1], ps[:, 0], 'red')

# Draw nodes on the image
node, nodes = graph._node, graph.nodes()
ps = np.array([node[i]['o'] for i in nodes])
plt.plot(ps[:, 1], ps[:, 0], 'g.')
plt.title('Jacob\'s Skeletonized Graph')

# Save the image with overlayed edges and nodes
plt.savefig('Overlay_Maze.jpg')
plt.show()

# Find the start and end points of the maze
start_point = None
end_point = None

# Check if there are any white pixels on the top and bottom rows
for i in range(img.shape[1]):
    if img[0][i][0] == 255:
        start_point = (0, i)
    if img[img.shape[0]-1][i][0] == 255:
        end_point = (img.shape[0]-1, i)

# Check if there are any white pixels on the left and right columns
for i in range(img.shape[0]):
    if img[i][0][0] == 255:
        start_point = (i, 0)
    if img[i][img.shape[1]-1][0] == 255:
        end_point = (i, img.shape[1]-1)

# If start or end points are not found, use top-left and bottom-right corners
if start_point is None:
    start_point = (0, 0)
if end_point is None:
    end_point = (img.shape[0]-1, img.shape[1]-1)

# Print start and end points
print(f"Start point: {start_point}")
print(f"End point: {end_point}")

# Find the shortest path from the start point to the end point
path, cost = shortest_path(start_point, end_point, threshold2)
path = np.array(path)

# Save the image with overlayed edges, nodes, and path
plt.imshow(img, cmap='gray')
plt.plot(path[:, 1], path[:, 0], 'b', linewidth=4, alpha=0.5, solid_capstyle='round')
plt.title('Skeletonize with path')
plt.savefig('Skeletonize_with_path_and_padding.jpg')
plt.show()
