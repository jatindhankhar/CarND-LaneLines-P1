import cv2
import numpy as np
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def otsu_method(img):
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    return [low_thresh,high_thresh]


def select_white(image):
    color_select  = np.copy(image)
    red_threshold = 190
    blue_threshold = 190
    green_threshold = 190
    rgb_threshold = [red_threshold,blue_threshold,green_threshold]
    # Identify pixels below the threshold
    thresholds = (image[:,:,0] < rgb_threshold[0]) \
            |(image[:,:,1] < rgb_threshold[1]) \
            |(image[:,:,2] < rgb_threshold[2])

    color_select[thresholds] = [0,0,0]
    
    return color_select

def pipeline(image):
    imshape = image.shape
    gray = grayscale(image)
    blur = gaussian_blur(image,5) # Blur with a kernel size of 5
    # Calculate high and low threshold using otsu's method 
    low_thresh,high_thresh = otsu_method(gray)
    # Apply Canny Edge Detection
    edges = canny(blur,low_thresh,high_thresh)
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    line_image = hough_lines(edges,2,1,30,10,7)
    res = weighted_img(line_image,image)
    mask = np.zeros_like(edges)
    vertices = np.array([[(0,imshape[0]),(450,320),(490,320),(imshape[1],imshape[0])]],dtype=np.int32)
    cv2.fillPoly(mask,vertices,ignore_mask_color)

    masked_edges = cv2.bitwise_and(edges,mask)
    lines = hough_lines(masked_edges,2,np.pi/360,15,90,150)
# Create binary color image
    color_edges = np.dstack((edges,edges,edges))
    line_edges = cv2.addWeighted(image,0.8,lines,1,0)
    return line_edges

cap = cv2.VideoCapture('solidWhiteRight.mp4')
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('window-name',pipeline(frame))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cap.destroyAllWindows()
