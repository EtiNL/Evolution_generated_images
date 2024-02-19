import numpy as np
import cv2 as cv
from draw_particules import Draw_particules

image_path = 'raw_data/vangogh2.jpg'
target_img = cv.imread(image_path, cv.IMREAD_COLOR)
target_img = cv.resize(target_img, (target_img.shape[1]//4,target_img.shape[0]//4))
display_img = np.copy(target_img)

drawing = False



def test(image_path):
    global drawing, target_img, display_img

    def nothing(x):
        pass

    def draw_circle(event,x,y,flags,param):
        global drawing, target_img, display_img
        radius = cv.getTrackbarPos('Brush radius','image')
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            test_img = np.copy(target_img)
            target_img = Draw_particules(target_img, test_img, np.array([x]), np.array([y]), np.array([radius]))
            display_img = np.copy(target_img)
        # elif event == cv.EVENT_MOUSEMOVE:
        #     if drawing == True:
        #         test_img = np.copy(target_img)
        #         target_img = cv.getTrackbarPos('Brush radius','image') = Draw_particules(target_img, test_img, np.array([x]), np.array([y]), np.array([radius]))
        # elif event == cv.EVENT_LBUTTONUP:
        #     display_img = np.copy(target_img)
        #     drawing = False
    # Create a black image, a window and bind the function to window


    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_circle)
    cv.createTrackbar('Brush radius', 'image',10,100,nothing)
    while(1):
        cv.imshow('image',display_img)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
        # radius = cv.getTrackbarPos('Brush radius','image')
    cv.destroyAllWindows()

if __name__=='__main__':
    test('raw_data/vangogh2.jpg')
