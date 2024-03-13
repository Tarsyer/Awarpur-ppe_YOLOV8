from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import os
from datetime import datetime
import time
from tarsyer import video_stream_queue as VSQ
from config import *
put_logo = False
video_write = False
show_image = True
from nms import *
from config import *
from motion_detect import motion_detection
from tarsyer.SSD_TfLite_Detector import *
from nms import non_max_suppression_fast


camera_vsq = VSQ.VideoStreamQueue(VIDEO_INPUT, CAMERA_DETAIL, CROPPING_COORD_PRI, SKIP_FRAME, CAMERA_NO)

camera_vsq.vsq_logger.info('VSQ in progress')
camera_vsq.start()
total_frame_count = camera_vsq.stream.get(cv2.CAP_PROP_FRAME_COUNT)
print('total_frame_count = {}'.format(total_frame_count))
camera_vsq.vsq_logger.info('CODE STARTED')

motion_confirm = False
motion_thresh_counter = 0
NO_OBJECT_THRESH = 3
prev_frame_time = time.time() - 5000
MOTION_CONFIRM_THRESH = 3

# defining obj detector function for person
object_detector = SSD_TfLite_Detection(CONFIDENCE_THRESH, MODEL_PATH)

if put_logo:
    # Define logo coordinates for the first logo
    LOGO_y1, LOGO_y2, LOGO_x1, LOGO_x2 = 100, 400, 100, 400

    # Read and resize the first logo
    LOGO1 = cv2.imread('logo.png', -1)
    LOGO1 = cv2.resize(LOGO1, (LOGO_x2 - LOGO_x1, LOGO_y2 - LOGO_y1))
    alpha_s1 = LOGO1[:, :, 3] / 255.0
    alpha_l1 = 1.0 - alpha_s1

model = YOLO("models/PH2-YOLOV8M.pt")
names = model.names
print(names)
# cap = cv2.VideoCapture("rtsp://admin:Admin%40123@192.168.10.66")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

save_image_dir = "/tmp/alert_images"
save_image_dir_for_email = "/tmp/alert_images_for_email"
if not os.path.exists(save_image_dir):
    os.mkdir(save_image_dir)
if not os.path.exists(save_image_dir_for_email):
    os.mkdir(save_image_dir_for_email)

# if video_write:
#     # Video writer
#     video_writer = cv2.VideoWriter("processed_awarpur_safety_with-white.avi",
#                                    cv2.VideoWriter_fourcc(*'mp4v'),
#                                    fps, (w, h))
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    # {0: 'hard_helmet', 1: 'jacket', 2: 'head', 3: 'suit'}
    #{0: 'suit_helmet', 1: 'suit', 2: 'gloves', 3: 'hand_without_gloves', 4: 'hard_helmet', 5: 'head', 6: 'jacket'}
    if label == 0:  # hard_helmet
        color = (145, 145, 255)

    elif label == 1:  # jacket
        color = (89, 189, 255)

    elif label == 2:  # hand_without_gloves
        color = (36, 36, 114)

    elif label == 3:  # gloves
        color = (177, 255, 131)

    elif label == 4:  # suit
        color = (6, 207, 58)

    elif label == 5:  # suit_helmet
        color = (0, 128, 33)

    elif label == 6:  # head
        color = (49, 49, 255)
    else:
        color = [200, 200, 200]
    return tuple(color)

consecutive_frames = {2: 0, 0: 0, 6: 0, 1: 0, 3: 0, 4: 0, 5: 0}  # classes: hand_without_gloves, hard_helmet, head, jacket
consecutive_frames_threshold = 1
skip_frames = 20
skip_counter = 0
frame_counter = 0
idx = 0
filter_class = [2, 0, 6, 1, 3, 4, 5] # classes: hand_without_gloves, hard_helmet, head, jacket
no_detection_output_counter = 0
while True:
    curr_time = time.time()
    camera_dict = camera_vsq.read()
    camera_status = camera_dict['camera_status']
    # ret, frame = video_cap.read()
    processing_start_time = time.time()
    if camera_status:
        resized_frame, big_frame = camera_dict['image']
        # im0 = im_ori[136:1520, 544:1928]
        im0 = resized_frame

        if curr_time - prev_frame_time > 60:
            prev_frame_time = curr_time
            base_frame = resized_frame.copy()
            print('saving previous frame.')
            camera_vsq.vsq_logger.info('Changing previous frame for motion detector')
        frame_counter += 1

        if not motion_confirm:
            cv2.putText(resized_frame, "Motion detector", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
            returned_status, _, _ = motion_detection(base_frame, resized_frame, MIN_CONTOUR_AREA)

            if returned_status:
                motion_thresh_counter += 1
                print("Motion Detected")

                if motion_thresh_counter == MOTION_CONFIRM_THRESH:
                    motion_thresh_counter = 0
                    motion_confirm = True
                    camera_vsq.vsq_logger.info('Universal: Motion Detected.')

            else:
                motion_thresh_counter = 0

        else:
            m_time = time.monotonic()
            motion_detect_start_time = time.monotonic()
            bbox_lists, class_lists, scores = object_detector.inference(resized_frame)
            print("Inference time: ", time.monotonic()-m_time)
            if len(bbox_lists) > 0:
                nms_list = non_max_suppression_fast(bbox_lists, 0.5)
            else:
                nms_list = []

            if len(bbox_lists) == 0:
                no_detection_output_counter += 1
                if no_detection_output_counter >= NO_OBJECT_THRESH:
                    no_detection_output_counter = 0
                    motion_confirm = False
                    base_frame = resized_frame.copy()
                    camera_vsq.vsq_logger.info('Universal: Motion confirm False')

            elif len(bbox_lists) > 0:
                print("Person Detector On")
                for class_name in class_lists:
                    person_centroid = (0, 0)
                    if class_name == 0:
                        cls_index = class_lists.index(class_name)
                        bboxes_person = bbox_lists[cls_index]
                        #cv2.rectangle(resized_frame, (bboxes_person[0], bboxes_person[1]), (bboxes_person[2], bboxes_person[3]), (0, 255, 0), 2)
                        person_centroid = (int((bboxes_person[0] + bboxes_person[2])/2), int((bboxes_person[1] + bboxes_person[3])/2))
                        # cv2.circle(resized_frame, person_centroid, 3, (0, 0, 255), 1)
                        pass
                    cv2.putText(resized_frame, "Person detector", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 100), 1)
                    motion_detect_start_time = time.monotonic()

                    inference_start_time = time.time()
                    results = model.predict(im0, show=False, iou=0.1, conf=0.1)
                    # print('Results')
                    # print(results)
                    inference_end_time = time.time()
                    inference_time = inference_end_time - inference_start_time
                    print(f"Inference Time: {inference_time} seconds")

                    boxes = results[0].boxes.xyxy.cpu().tolist()
                    clss = results[0].boxes.cls.cpu().tolist()
                    print(clss)

                    total_cls = []
                    if boxes is not None:
                        for box, cls in zip(boxes, clss):
                            idx += 1
                            color = compute_color_for_labels(cls)
                            # cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2, )
                            cls = int(cls)
                            # Check if the detected class is one of interest
                            if cls in filter_class:
                                if cls in consecutive_frames:
                                    consecutive_frames[cls] += 1
                                    if consecutive_frames[cls] >= consecutive_frames_threshold:

                                        cy = (box[1]+box[3])/2
                                        if cy > 80: # ignoring top area of camera/ ignoring far person 
                                            total_cls.append(cls)

                                            cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2, )
                                            # Save image with datetime
                                            print(cls)
                                        consecutive_frames[cls] = 0  # Reset the counter after saving the image
                                else:
                                    # Reset the counter if the detected class is not of interest
                                    consecutive_frames[cls] = 0

                    helmet_flag = 2
                    suit_flag = 2
                    gloves_flag = 2
                    print(f'total_cls {total_cls}')


                    if len(total_cls) > 0:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        # {0: 'hard_helmet', 1: 'jacket', 2: 'head', 3: 'suit'}
                        # elif label == 3:  # gloves

                        # elif label == 4:  # suit

                        # elif label == 5:  # suit_helmet
                        if 5 in total_cls:
                            helmet_flag = 1
                        if 3 in total_cls:
                            suit_flag = 1

                        if 0 in total_cls or 2 in total_cls:
                            helmet_flag = 0
                        if 1 in total_cls:
                            suit_flag = 0

                        # if cls == 2:
                        #     image_name = f"{timestamp}_WRHS_110.jpg"
                        # elif cls == 0:
                        #     image_name = f"{timestamp}_WRHS_011.jpg"
                        # elif cls == 6:
                        #     image_name = f"{timestamp}_WRHS_011.jpg"
                        # elif cls == 1:
                        #     image_name = f"{timestamp}_WRHS_101.jpg"

                        if helmet_flag == 0 or suit_flag == 0 or gloves_flag == 0:

                            image_name = f"{timestamp}_PH2_{helmet_flag}{suit_flag}{gloves_flag}.jpg"

                            filename = f"{save_image_dir}/{image_name}"
                            filename_for_email = f"{save_image_dir_for_email}/{image_name}"

                            cv2.imwrite(filename, big_frame)
                            cv2.imwrite(filename_for_email, big_frame)

        if put_logo:
            # Blend the first logo onto the frame
            for c in range(0, 3):
                im0[LOGO_y1:LOGO_y2, LOGO_x1:LOGO_x2, c] = (
                        alpha_s1 * LOGO1[:, :, c] + alpha_l1 * im0[LOGO_y1:LOGO_y2, LOGO_x1:LOGO_x2, c]
                )
        if show_image:
            cv2.namedWindow('resized_frame', cv2.WINDOW_KEEPRATIO)
            cv2.imshow("resized_frame", im0)


        processing_end_time = time.time()
        processing_time = processing_end_time - processing_start_time
        print(f"processing Time: {processing_time} seconds")
        # if video_write:
        #     video_writer.write(im0)

        if show_image:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# cap.release()
# if video_write:
#     video_writer.release()
if show_image:
    cv2.destroyAllWindows()

