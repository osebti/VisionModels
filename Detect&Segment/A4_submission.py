
import numpy as np

from PIL import Image
import numpy as np
import subprocess
import sys




subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
from ultralytics import YOLO




def detect_and_segment(images):
    """

    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """

    N = images.shape[0]

        # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)
    # pred_seg: Your predicted segmentation for the image, shape [N, 4096]
    pred_seg = np.empty((N, 4096), dtype=np.int32)



    model_det=YOLO("best_det_final.pt")
    model_seg=YOLO("best_seg_final.pt")

    for i in range(N):
        # convert array to image
        img=images[i].reshape((64,64,3))
        img=Image.fromarray(img)
        result=model_det(img,verbose=False)[0]
        # get boxes and classification
        bboxes=result.boxes.xyxy.cpu()
        bboxes=bboxes.numpy().astype(np.float64)

        classes=result.boxes.cls.cpu().numpy().astype(np.int32)

        if classes.size==2 and int(classes[0])<=int(classes[1]):
          row1 =[bboxes[0][1],bboxes[0][0],bboxes[0][3],bboxes[0][2]]
          row2 =[bboxes[1][1],bboxes[1][0],bboxes[1][3],bboxes[1][2]]
          pred_class[i]=classes

        elif classes.size==2 and int(classes[0])>int(classes[1]):
          row2 =[bboxes[0][1],bboxes[0][0],bboxes[0][3],bboxes[0][2]]
          row1= [bboxes[1][1],bboxes[1][0],bboxes[1][3],bboxes[1][2]]
          pred_class[i][0]=classes[1]
          pred_class[i][1]=classes[0]

        else:
          try:
            row1 =[bboxes[0][1],bboxes[0][0],bboxes[0][3],bboxes[0][2]]
            row2 = np.zeros((4))
          except:pass

        pred_bboxes[i][0]=np.array(row1)
        pred_bboxes[i][1]=np.array(row2)




        mask=np.full((64,64),10)
        result=model_seg(img,verbose=False)[0]
        masks=result.cpu().masks
        if masks!=None:

          labels=result.boxes.cls.cpu().numpy().astype(np.int32)

          y=0
          masks=masks.xy
          for res in masks:
              for point in res:
                  mask[np.int32(point[0])][np.int32(point[1])]=labels[y]
              y+=1

          mask=mask.reshape((4096))
          pred_seg[i]=mask

        else:
          mask=mask.reshape((4096))
          pred_seg[i]=mask


    # add your code here to fill in pred_class and pred_bboxes


    return pred_class, pred_bboxes, pred_seg
