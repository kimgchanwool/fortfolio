import cv2
import numpy as np
import os

def tracking(inp): #비디오 파일 열기
  video_path = f'{inp}.mp4' #경로 지정
  vid = cv2.VideoCapture(video_path) #경로에 있는 비디오 불러오기

  output_size = (375, 667) # (width, height) # 너비 높이 지정
  fit_to = 'height' #높이로 지정

  #비디오 내보낼때 out을 설정함.
  fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #mp4파일 저장을 위한 코덱, 압축방식, 색상, 포맷등이다.
  out = cv2.VideoWriter('%s_output.mp4' % (video_path.split('.')[0]), fourcc, vid.get(cv2.CAP_PROP_FPS), output_size) #video writer은 동영상 저장하기 %s에 man이라는 이름이 들어감.man_out으로 이름 저장함.
  #videowriter의 기본 적인 틀 retval = cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None)
  #파일이름, 방식, 초당 프레임수, 프레임 크기, 색상(none은 흑백 true는 컬러 default는 true) 현재로 봤을때 color는 true로 디폴트값이다. cv2.CAP_PROP_FPS는 현재 프레임 속도다.

  # check file is opened
  if not vid.isOpened(): #isopened는 파일이 열렸는지 확인
    exit() #열리지 않았다면 false가 나왔다면 끝내기.

  # initialize tracker 튜플을 만들어 골라사용.
  OPENCV_OBJECT_TRACKERS = { #opencv에서 tracker 오픈소스 api가 제공된다. 정확성이 높을수록 속도가 빨라서 csrt의 경우 연산은 느리지만 정확도과 꽤 높아서 csrt사용 자세한 내용은 논문이라 찾기 힘들다 ㅠ
    "csrt": cv2.TrackerCSRT_create,
    #"kcf": cv2.TrackerKCF_create,
    #"boosting": cv2.TrackerBoosting_create,
    #"mil": cv2.TrackerMIL_create,
    #"tld": cv2.TrackerTLD_create,
    #"medianflow": cv2.TrackerMedianFlow_create,
    #"mosse": cv2.TrackerMOSSE_create
  }

  tracker = OPENCV_OBJECT_TRACKERS['csrt']() #트래커 모델에 csrt트래커 모델을 넣기.

  # global variables
  top_bottom_list, left_right_list = [], []

  # main
  ret, img = vid.read()

  cv2.namedWindow('Select Window') #해당 이름을 가진 윈도우 창을 생성한다. #어차피 imshow에서 이름을 지정해줘서 굳이 없어도 되는 듯.
  cv2.imshow('Select Window', img) #해당 이름의 윈도우 창에 img를 보여준다.

  # select ROI
  rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True) #roi사각형 받기 ROI는 관심영역을 지정하는 범위. 즉, 트래킹 대상이 될 범위
  cv2.destroyWindow('Select Window') #rect에 범위 저장후 지정된 윈도우 창을 파괴

  # initialize tracker
  tracker.init(img, rect) #객체를 생성후 roi사각형을 받아서 해당 객체를 트래킹하도록 트래커를 발동.

  while True:
    ret, img = vid.read() #ret은 false true값으로 없으면 true 있으면 false의 의미. img는 영상을 받아옴.

    if not ret: #영상이 없으면 true라서 true면 끝.
      exit()

    
    success, box = tracker.update(img) #정보 업데이트하는 중. init된 것을 계속 업데이트해서 객체를 추적 반환 값으로 해당 좌표를 반환. box에는 좌표가 담긴다. success는 성공여부인듯.
    left, top, w, h = [int(v) for v in box] #좌표를 소분화한다.
    right = left + w #좌측 + 너비
    bottom = top + h # 위쪽 + 높이

    top_bottom_list.append(np.array([top, bottom])) #top, bottom꼴로 묶어 만든 numpy배열 추가
    left_right_list.append(np.array([left, right])) #left, right꼴로 묶어 만든 numpy배열 추가
    #공부 필요!


    if len(top_bottom_list) > 10: # 윈도우 길이보다 배열로 나타내야할 길이가 10보다 크면 첫번째 배열을 삭제한다.
      del top_bottom_list[0]
      del left_right_list[0]

    avg_height_range = np.mean(top_bottom_list, axis=0).astype(int) #평균을 내는 것임. 탑과 바텀의 평균을 정수로
    avg_width_range = np.mean(left_right_list, axis=0).astype(int) #좌측과 우측의 평균을 정수로
    avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) # (x, y) #너비평균과 높이 평균을 추가 즉 중심임.

    scale = 2.0 #클수록 많이 보여줌.
    avg_height = (avg_height_range[1] - avg_height_range[0]) * scale #높이 평균이다 탑에서 바텀을 빼서 scale을 곱해서 높이평균을 만듦             그리고 scale은 트래킹된 영상을 보여줄 건데 해당 ROI에서 얼마만큼 더 보여줄 것인지 정하는것 넉넉히 2.0으로가봄
    avg_width = (avg_width_range[1] - avg_width_range[0]) * scale #너비평균 우측에서 좌측을 뺀 것에 scale을 곱해서 너비평균을 만듦

    avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2]) #중심에서 원하는 높이의 절반을 빼고 절반을 더하면 각각 상단 하단으로 새로운 height 배열이 만들어짐
    avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2]) #중심에서 원하는 너비의 절반을 빼고 절반을 더하면 각각 좌측 우측으로 width 배열이 만들어짐

    if fit_to == 'width': #는 무조건 height
      avg_height_range = np.array([
        avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
        avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
      ]).astype(int).clip(0, 9999)

      avg_width_range = avg_width_range.astype(int).clip(0, 9999)
    elif fit_to == 'height': #아마 여기 통과할 듯
      avg_height_range = avg_height_range.astype(int).clip(0, 9999) #간단하게 삑사리 방지용으로 0이하값은 0으로 9999이상값은 9999로 min하고 max를 정해두는 것임. 범위 벗어나지 않게 하려는 것.

      avg_width_range = np.array([
        avg_center[0] - avg_height * output_size[0] / output_size[1] / 2, #평균너비에다가 (평균 높이에 처음 정한 너비대 높이 비율을 곱하고 /2해준 것을 뺀 것이 left
        avg_center[0] + avg_height * output_size[0] / output_size[1] / 2 #이건 반대로 더한것이 right
      ]).astype(int).clip(0, 9999) #에서 삑사리 방지.#마찬가지지만 일단 width범위를 다시 지정.

    result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy() #불필요한 부분 잘라내기. 이미지에서 너비(left부터 right)와 높이(top부터 bottom)만큼 가져와서 copy안하면 같은객체가 된다.
    result_img = cv2.resize(result_img, output_size) #result_img를 output에 맞게 사이즈 재지정

    pt1 = (int(left), int(top))
    pt2 = (int(right), int(bottom))
    cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)

    cv2.imshow('img', img) #원본 img 보여주기
    cv2.imshow('result', result_img) #result(범위 줄어들은 것)보여주기
    out.write(result_img)
    #아마 여기서 아스키 코드로 전환할 수 있을 듯.
    #두개를 같이 보여주면서 얼마나 범위 줄어들었나 보여주는 것임.
    #write video
    if cv2.waitKey(1) == ord('q'): #해당 키 누르면 끝내기.
      break

  vid.release() #cap객체 해제 윈도우 창 닫기위해서
  out.release() #out객체 해제 윈도우 창 닫기위해서
  cv2.destroyAllWindows() #모든 영상 보여줬으면 닫기.

def ascii(inp):
  nw = 100

  CHARS = ' .,-_`~+=*^)(&%$#@'
  CHARS = sorted(CHARS, reverse=True)

  vid = cv2.VideoCapture(f'{inp}_output.mp4')
  os.system('cls')
  while vid.isOpened:
    ret, img = vid.read()
    if not ret:
      break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    nh = int(h / (2 * w) * nw)

    img = cv2.resize(img, (nw, nh))

    for row in img:
      for pixel in row:
        i = int(pixel / 256 * len(CHARS))
        print(CHARS[i], end='')
      print()
    os.system('cls')

if __name__ == '__main__':
  #inp = input('what r u want tracking?(only name plz, not .mp4...) : ')
  #tracking(inp)
  ascii('man')
