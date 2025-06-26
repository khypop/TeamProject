import cv2


cap=cv2.VideoCapture(0)                         # 0번 카메라에 연결

if cap.isOpened():                  
    while True:
        ret, img=cap.read()                     # 카메라를 읽음
        if ret:
            cv2.imshow('camera',img)            # 이미지를 표시
            if cv2.waitKey(10) != -1:           # 10ms동안 키 입력을 대기 그후 다시 카메라 읽기
                cv2.imwrite('photo.jpg', img)   # 키가 입력되면 사진을 저장(현재 스크립트가 있는 폴더에 저장)
                break                           # 키가 입력되면 중지

else:
    print("can't open camera")
cap.release()
cv2.destroyAllWindows()