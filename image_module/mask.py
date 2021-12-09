import cv2, random, os
import numpy as np

def is_rectangle_overlap(rectangle, answers, margin = 0):
    """
    Check if the input rectangle is overlapped.
    """
    x1, y1, w1, h1 = rectangle
    for x2, y2, w2, h2 in answers:
        if not (x2 > x1 + w1 + margin or x1 > x2 + w2 + margin or y2 > y1 + h1 + margin or y1 > y2 + h2 + margin):
            return True
    
    return False

def create_mask(img):
    gray_img = cv2.cvtColor(np.array(img) , cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    high_thresh, thresh_im = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    gray_img = cv2.Canny(gray_img, 0.5 * high_thresh, high_thresh)
    contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    li, answers = [], []

    for i, (same_cnt, prev_cnt, child_cnt, parent_cnt) in enumerate(hierarchy[0]):
        if child_cnt != -1:
            x1, y1, w1, h1 = cv2.boundingRect(contours[i])
            current_area = w1 * h1 
            current_ratio = float(w1) / h1

            if 0.3 <= current_ratio <= 3 and 1000 <= current_area <= 10000:
                li.append((x1, y1, w1, h1))

    while (len(answers) < 5 and len(li)):
        index = random.randint(0, len(li) - 1)
        x, y, w, h = li.pop(index)
        if is_rectangle_overlap((x, y, w, h), answers, margin=20): 
            continue
        
        answers.append((x, y, w, h))

    return answers
