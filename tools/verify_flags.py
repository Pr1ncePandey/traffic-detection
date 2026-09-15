import cv2
import numpy as np
import pandas as pd

df = pd.read_csv("outputs/full_lane.csv")
cap = cv2.VideoCapture("samples/short/input.mp4")
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
for fno in [1000, 1050]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, img = cap.read()
    assert ret
    cv2.polylines(img, [np.array([[0, 0], [int(.34 * W), 0],
                                  [int(.05 * W), H], [0, H]], np.int32)],
                  True, (255, 255, 0), 2)
    d = df[df.frame == fno]
    for _, r in d.iterrows():
        color = (0, 0, 255) if "wrong_way" in str(r.lane_flag) else (0, 255, 0)
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
        cv2.putText(img, "#%d %s" % (int(r.object_id), r.lane_id),
                    (int(r.x1), max(0, int(r.y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite("outputs/verify_%d.jpg" % fno, img)
    print("saved", fno, len(d), "boxes")
