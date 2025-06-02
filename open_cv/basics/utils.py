import cv2 as cv

def img_resize(img, scale=0.5):
    '''
    Resizes the size of the image passed.

    img: is the image frame read
    scale: is the scaling specified

    Return: returns the resized image

    '''

    width = int(img.shape[0] * scale)
    height = int(img.shape[1] * scale)

    dimensions = (width, height)
    
    resized_img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

    return resized_img
    

def video_resize(frame, scale=0.5):

    '''
    resized the size of a video

    frame: passed frame for resizing.
    scale: scaling factor.

    Return: resized video frame
    '''

    width = int(frame.shape[0] * scale)
    height = int(frame.shape[1] * scale)

    dimensions = (width, height)
    resized_frame = cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

    return resized_frame
