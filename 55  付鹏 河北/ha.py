import cv2
def haxi(img1):
    ing = cv2.resize(img1, (8, 8), interpolation=cv2.INTER_CUBIC)
    ing = cv2.cvtColor(ing, cv2.COLOR_BGR2GRAY)
    s = 0
    a=''
    for i in range(8):
        for j in range(8):
            s = s + ing[i][j]
    z = s * 1.0 / (64)
    for i in range(8):
        for j in range(8):
            if ing[i][j] < z:
                a= a+'0'
            else:
                a = a+'1'
    
    return a


def haxi1(img2):
    ing = cv2.resize(img2, (9, 8), interpolation=cv2.INTER_CUBIC)
    ing = cv2.cvtColor(ing, cv2.COLOR_BGR2GRAY)
    s=''
    for i in range(8):
        for j in range(8):
            if ing[i][j] < ing[i][j - 1]:
                s = s+'0'
            else:
                s = s+'1'
    return s

def compare(a,b):
    c=0
    for i in range(len(a)):
        if a[i]!=b[i]:
            c=c+1
    return c

img1 = cv2.imread('F:\\lenna.png')
img2 = cv2.imread('F:\\lenna_noise.png')
a= haxi(img1)
b = haxi1(img2)
c= compare(a,b)
print(a, b, sep='\n')
print(c)
