import cv2
import numpy as np

def extract_minutiae(fingerprint_image_path):
    # Read the image in grayscale
    img = cv2.imread(fingerprint_image_path, cv2.IMREAD_GRAYSCALE)

    
    # Apply adaptive thresholding to convert the image to binary
    ret, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Compute the thinning of the image to get a skeletonized version
    skeleton = cv2.ximgproc.thinning(img_bin)
    
    # Find minutiae by computing the (3, 3) cross of the skeletonized image
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    minutiae = cv2.morphologyEx(skeleton, cv2.MORPH_HITMISS, kernel)
    
    return minutiae

def match_minutiae(minutiae_img1, minutiae_img2):
    # Find the coordinates of non-zero points from the minutiae images
    coords_img1 = np.column_stack(np.where(minutiae_img1 > 0))
    coords_img2 = np.column_stack(np.where(minutiae_img2 > 0))

    # Set a threshold for matching points. If two minutiae points are closer
    # than this threshold, they are considered a match.
    threshold_distance = 18# you can tweak this value

    match_count = 0
    print(len(coords_img1))
    for point1 in coords_img1:
        for point2 in coords_img2:
            distance = np.linalg.norm(point1 - point2)
            if distance < threshold_distance:
                match_count += 1
                break  # Break out of the inner loop if a match is found for the current point
    print(match_count*100/len(coords_img1))

    # A simple criterion: if more than 70% of the minutiae match, consider the fingerprints a match.
    # This is a very naive criterion and you can refine it.
    if match_count > 0.8 * len(coords_img1):
        return True
    else:
        return False

# Testing the matching function
if __name__== '__main__':
    fingerprint_image_path1 = 'E:\\Fingerprint-Matching\\G_final.png'
    fingerprint_image_path2 = 'E:\\Fingerprint-Matching\\Cm_final.png'  # Another fingerprint image for comparison
    minutiae_img1 = extract_minutiae(fingerprint_image_path1)
    minutiae_img2 = extract_minutiae(fingerprint_image_path2)
    
    
    is_match = match_minutiae(minutiae_img1, minutiae_img2)
    if is_match:
        print("The fingerprints match!")
    else:
        print("The fingerprints do not match.")