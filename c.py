import cv2
import numpy as np


# 回调函数
def nothing(pos):
    pass


if __name__ == '__main__':
    img = cv2.imread('test.jpg')
    height = 256
    width = int(img.shape[0] * height / img.shape[1])
    img = cv2.resize(img, (width, height))
    img = cv2.GaussianBlur(img, (9, 9), 0)

    # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # f = np.fft.fft2(img2)
    # fshift = np.fft.fftshift(f).real
    # magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # cv2.imshow("fft", magnitude_spectrum)
    # cv2.waitKey(0)

    edges = img
    cv2.namedWindow('Original')
    cv2.namedWindow('Canny')
    cv2.createTrackbar('Min', 'Canny', 0, 100, nothing)
    cv2.createTrackbar('Max', 'Canny', 100, 200, nothing)
    cv2.setTrackbarPos("Min", "Canny", 100)
    cv2.setTrackbarPos("Max", "Canny", 200)

    while True:
        min = cv2.getTrackbarPos('Min', 'Canny')
        max = cv2.getTrackbarPos('Max', 'Canny')
        edges = cv2.Canny(img, min, max)
        frame = np.zeros((500, 500), np.uint8)
        frame[:edges.shape[0], :edges.shape[1]] = edges
        print(np.sum(edges == 255))
        cv2.imshow('Original', img)
        cv2.imshow('Canny', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
