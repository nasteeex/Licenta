import os
from os import listdir
from os.path import isfile, join
from datetime import datetime
from time import perf_counter
import numpy as np
import pandas as pd
import easyocr
import cv2
import tensorflow as tf
import csv
import uuid
import smtplib
import ssl
from email.message import EmailMessage
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from google.protobuf import text_format
import shutil
from threading import Thread
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----- 0. Setup of Paths

# File name, links that will be used frequently
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'https://download.tensorflow.org/models/object_detection/tf2/20200711' \
                       '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz '
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# Paths that will be used frequently
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('Tensorflow', 'protoc'),
    'REALTIMEDETECTIONS_PATH': os.path.join('RealTimeDetections')
}

# Files that will be used frequently
files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# ----- 1. Downloading TF Models Pretrained Models from Tensorflow Model Zoo and Installing TFOD
# ----- 2. Creating the Label Map
# ----- 3. Creating TF records
# ----- 4. Copying Model Config to Training Folder

# ----- 5. Updating Config For Transfer Learning

# Getting configs from pipeline config file
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

# Read contents of pipeline config file
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Write updated configs to pipeline config file
config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)

# ----- 6. Training the model
# ----- 7. Evaluating the Model

# ----- 8. Loading Trained Model From Checkpoint and Building a Detection Model

# Preventing GPU from complete consumption of resources
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RunTimeError as err:
        print(err)

# Loading pipeline config file
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restoring latest checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()


# Building a detection model
@tf.function
def detect_fn(image):
    """Build detection model function

        Parameters:
            image (tensorflow.python.framework.ops.EagerTensor): tensorflow image to detect plate on

        Returns: detections_local (dict, dict_keys(['detection_boxes', 'detection_scores', 'detection_classes', 
        'raw_detection_boxes', 'raw_detection_scores', 'detection_multiclass_scores', 'detection_anchor_indices', 
        'num_detections'])): detected plate """

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections_local = detection_model.postprocess(prediction_dict, shapes)
    return detections_local


# ----- 9. Detecting plate from an Image, OCR, Saving Results

# --- 9. a) Detecting plate from an Image

# Categories extracted from labelmap
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# Image to detect plate number from
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'Cars1.png')

# Preprocessing image
img_orig = cv2.imread(IMAGE_PATH)
print('ORIGINAL')
plt.imshow(img_orig)
plt.show()

img_resized = cv2.resize(img_orig, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
print('IMPROVED RESOLUTION')
plt.imshow(img_resized)
plt.show()

img_detailsEnhanced = cv2.detailEnhance(img_resized, sigma_s=10, sigma_r=0.15)
print('DETAILS ENHANCED')
plt.imshow(img_detailsEnhanced)
plt.show()

img_BilateralFilterEnhanced = cv2.bilateralFilter(img_detailsEnhanced, 9, 75, 75)
print('BILATERAL FILTER ENHANCED')
plt.imshow(img_BilateralFilterEnhanced)
plt.show()

img_BilateralFilterEnhancedGray = cv2.cvtColor(img_BilateralFilterEnhanced, cv2.COLOR_BGR2GRAY)
print('BILATERAL GRAY ENHANCED')
plt.imshow(img_BilateralFilterEnhancedGray, cmap='gray')
plt.show()

img_threshEnhanced = \
    cv2.threshold(cv2.medianBlur(img_BilateralFilterEnhancedGray, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
print('THRESH 1 ENHANCED')
plt.imshow(img_threshEnhanced, cmap='gray')
plt.show()

img_backtorgb = cv2.cvtColor(img_threshEnhanced, cv2.COLOR_GRAY2RGB)

image_np = np.array(img_orig)

# Detection
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# Detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

# Visualizing detection
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'] + label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=.8,
    agnostic_mode=False)

print('Detected plate')
plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()


# --- 9. b) OCR Filtering


# Filter OCR function
def filter_text(region, ocr_result, region_threshold_local) -> list:
    """Filter OCR function

        Parameters:
            region (numpy.ndarray): region of interest for plate detection
            ocr_result (list): OCR characters
            region_threshold_local (float): threshold for region to run OCR over

        Returns:
            plate (list): plate characters"""

    rectangle_size = region.shape[0] * region.shape[1]

    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold_local:
            plate.append(result[1])
    return plate


# --- 9. c) OCR proccess and filtering

# Thresholds for detection and OCR filtering
detection_threshold = 0.7
region_threshold = 0.6


# OCR function
def ocr_it(image, detections_local_, detection_threshold_local, region_threshold_local_):
    """OCR function

        Parameters: image (tensorflow.python.framework.ops.EagerTensor): tensorflow image to detect plate on 
        detections_local_ (dict, dict_keys(['detection_boxes', 'detection_scores', 'detection_classes', 
        'raw_detection_boxes', 'raw_detection_scores', 'detection_multiclass_scores', 'detection_anchor_indices', 
        'num_detections'])): detected plate: detection_threshold_local (float): threshold for detection to be 
        considerred successfull region_threshold_local_ (float): threshold for region to run OCR over 

        Returns:
            text (list): plate characters
            region (tensorflow.python.framework.ops.EagerTensor): tensorflow image, detected plate region """

    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x > detection_threshold_local, detections['detection_scores']))
    boxes = detections_local_['detection_boxes'][:len(scores)]
    # classes = detections_local_['detection_classes'][:len(scores)]

    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)

        text = filter_text(region, ocr_result, region_threshold_local_)

        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.show()
        print(text)
        return text, region


# --- 9. d) Saving Results

# Saving detections
def save_results(text, region):
    """Save results function

        Parameters:
            text (list): plate characters
            region (tensorflow.python.framework.ops.EagerTensor): tensorflow image, detected plate region

        Returns: N/A"""

    now = datetime.now()
    date_string = now.strftime('%y-%m-%d')
    print(date_string)
    time_string = now.strftime('%H-%M-%S')
    print(time_string)

    folder_path_images = 'RealTimeDetectionsImages\\RealTimeDetections-{}'.format(date_string)
    img_name = '{}_{}.jpg'.format(time_string, uuid.uuid1())
    # forlder_path_csv = 'RealTimeDetections'
    csv_filename = 'RealTimeDetections\\RealTimeDetections-{}.csv'.format(date_string)

    cv2.imwrite(os.path.join(folder_path_images, img_name), region)

    with open(csv_filename, mode='a', newline='') as f_loc:
        csv_writer = csv.writer(f_loc, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if text == []:
            text = ['unreadable']
        csv_writer.writerow([date_string, time_string, img_name, text])


# ----- 10. Real Time Detections from Webcam

# ----- 11. Getting and processing data from the CSV detection files
# --- 11. a) Getting data from CSV file

def get_data_from_csv(elem):
    """Get data from CSV function

        Parameters:
            elem (*.csv file): CSV file

        Returns: [list_date_new, list_time_new, list_file_name_new, list_plate_new] (list): lists of detection dates, 
        timestamps, filenames, plates """

    data_date = pd.read_csv(os.path.join('RealTimeDetections', elem), header=None, usecols=[0])
    data_time = pd.read_csv(os.path.join('RealTimeDetections', elem), header=None, usecols=[1])
    data_file_name = pd.read_csv(os.path.join('RealTimeDetections', elem), header=None, usecols=[2])
    data_plate = pd.read_csv(os.path.join('RealTimeDetections', elem), header=None, usecols=[3])

    list_date = data_date.values.tolist()
    list_time = data_time.values.tolist()
    list_file_name = data_file_name.values.tolist()
    list_plate = data_plate.values.tolist()

    list_date_new = [elem[0] for elem in list_date]
    list_time_new = [elem[0] for elem in list_time]
    list_file_name_new = [elem[0] for elem in list_file_name]
    list_plate_new = [elem[0] for elem in list_plate]

    list_plate_new_new = []
    for elem in list_plate_new:
        elem = elem[2:-2]
        list_plate_new_new.append(elem)

    # print(list_date_new)
    # print()
    # print(list_time_new)
    # print()
    # print(list_file_name_new)
    # print()
    # print(list_plate_new_new)
    # print()
    # print("--------------------------------------------------------")

    return [list_date_new, list_time_new, list_file_name_new, list_plate_new]


# --- 11. b) Redundancy plate process

def redundancy_function(dates_to_pass, timestamps_to_pass, filenames_to_pass, plates_to_pass):
    """Build detection model function Parameters: dates_to_pass (list): detection dates timestamps_to_pass (list): 
    detection timestamps filenames_to_pass (list): detection filenames plates_to_pass (list): detection plates 
    Returns: [timestamps_valids, dates_valids, filenames_valids, plates_valids] (list): lists of detection dates, 
    timestamps, filenames, plates """

    dates_valids = []
    timestamps_valids = []
    filenames_valids = []
    plates_valids = []

    for i, timestampi in enumerate(timestamps_to_pass):
        dates_valid = []
        timestamps_valid = []
        filenames_valid = []
        plates_valid = []
        only_one = True

        timestamp1 = (datetime.strptime(timestampi, "%H-%M-%S").time())

        flag = True
        if len(timestamps_valids) != 0:
            for elem in timestamps_valids:
                if timestamp1.strftime("%H-%M-%S") in elem:
                    flag = False
                    break

        if timestampi != timestamps_to_pass[-1] and flag is True:
            for j, timestampj in enumerate(timestamps_to_pass[i + 1:]):
                timestamp2 = (datetime.strptime(timestampj, "%H-%M-%S").time())
                # print(f'{timestamp1}, {timestamp2}')
                if (timestamp1.hour == timestamp2.hour) and (timestamp1.minute == timestamp2.minute):
                    if timestamp1.strftime("%H-%M-%S") not in timestamps_valid:
                        timestamps_valid.append(timestamp1.strftime("%H-%M-%S"))
                        dates_valid.append(dates_to_pass[i])
                        filenames_valid.append(filenames_to_pass[i])
                        plates_valid.append(plates_to_pass[i])
                    if timestamp2 not in timestamps_valid:
                        timestamps_valid.append(timestamp2.strftime("%H-%M-%S"))
                        dates_valid.append(dates_to_pass[i + 1 + j])
                        filenames_valid.append(filenames_to_pass[i + 1 + j])
                        plates_valid.append(plates_to_pass[i + 1 + j])
                        only_one = False
                elif only_one is True:
                    timestamps_valid.append(timestamp1.strftime("%H-%M-%S"))
                    dates_valid.append(dates_to_pass[i])
                    filenames_valid.append(filenames_to_pass[i])
                    plates_valid.append(plates_to_pass[i])
                    break
                else:
                    break

        if len(timestamps_valid) != 0:
            timestamps_valids.append(timestamps_valid)
            dates_valids.append(dates_valid)
            filenames_valids.append(filenames_valid)
            plates_valids.append(plates_valid)

    # print(timestamps_valids)
    # print()
    # print(dates_valids)
    # print()
    # print(filenames_valids)
    # print()
    # print(plates_valids)
    # print()

    return [timestamps_valids, dates_valids, filenames_valids, plates_valids]


# --- 11. c) Redundancy plate text process

def process_plate(strings_unreadable, dates, timestamps, filenames):
    """Redundancy function

        Parameters:
            strings_unreadable (list): detected plates
            dates (list): detected dates
            timestamps (list): detected timestamps
            filenames (list): detected filenames

        Returns:
        string_final (str): processed plate"""

    # print(f'Placute neprocesate:\n{strings_unreadable}\n')

    break_flag = False

    strings = []
    for index, elem in enumerate(strings_unreadable):
        if elem != "['unreadable']":
            strings.append(elem)

    counts_unreadable = []
    for elem in strings_unreadable:
        counts_unreadable.append(strings_unreadable.count(elem))
    # print(counts_unreadable)

    maximum_unreadable = max(counts_unreadable)
    if maximum_unreadable != "['unreadable']":
        maximum_index_unreadable = counts_unreadable.index(maximum_unreadable)
        date_kept = dates[maximum_index_unreadable]
        timestamp_kept = timestamps[maximum_index_unreadable]
        filename_kept = filenames[maximum_index_unreadable]
    else:
        from numpy.random.mtrand import random
        maximum_index_unreadable = random.randrange(0, len(counts_unreadable))
        date_kept = dates[maximum_index_unreadable]
        timestamp_kept = timestamps[maximum_index_unreadable]
        filename_kept = filenames[maximum_index_unreadable]

    # print(f'Numar de placute: {len(strings)}\n')
    # print(f'Placute:\n{strings}\n')

    strings_sorted = strings
    strings_sorted.sort()
    # print(f'Placute sortate:\n{strings_sorted}\n')

    strings_unique = set(strings)
    strings_unique = list(strings_unique)
    # print(f'Placute fara dubluri:\n{strings_unique}\n')

    if len(strings_unique) == 0:
        break_flag = True

    if break_flag:
        string_final = strings_unreadable[0]
        date_kept = dates[0]
        timestamp_kept = timestamps[0]
        filename_kept = filenames[0]

    if not break_flag:
        counts = []
        for elem in strings_unique:
            counts.append(strings.count(elem))
        # print(f'Numar de aparitii fiecare placuta unica:\n{counts}\n')

        maximum = max(counts)
        maximum_index = counts.index(maximum)
        # print(f'Cel mai mare numar de aparitii: {maximum}')
        # print(f'Se afla la indexul: {maximum_index}\n')

        strings_values = []
        strings_counts = []
        if counts.count(maximum) == 1:
            for index, elem in enumerate(strings_unique):
                if len(elem) == len(strings_unique[maximum_index]):
                    strings_values.append(elem)
                    strings_counts.append(counts[index])
                else:
                    pass
                    # print(f'{elem} stearsa din detectii!')
        elif counts.count(maximum) == len(strings_unique):
            for index, elem in enumerate(strings_unique):
                if len(elem) == len(strings_unique[0]):
                    strings_values.append(elem)
                    strings_counts.append(counts[index])
        else:
            for index, elem in enumerate(strings_unique):
                if len(elem) == len(strings_unique[maximum_index]):
                    strings_values.append(elem)
                    strings_counts.append(counts[index])

        # print(f'\nPlacute luate in considerare:\n{strings_values}')
        # print(f'Cu numerele de apartii:\n{strings_counts}\n')

        litere_posibile_placuta = []
        factori_placuta = []
        for i in range(0, len(strings_unique[maximum_index])):
            litere_posibile = []
            factori = []
            for index, elem in enumerate(strings_values):
                # print(f'Litera de pe indexul {i} din placuta cu numarul {index} este {elem[i]} si are un factor de 
                # {strings_counts[index]}') 
                litera_posibila = elem[i]
                factor = counts[index]
                if litera_posibila not in litere_posibile:
                    litere_posibile.append(litera_posibila)
                    factori.append(factor)
                else:
                    index_litera_existenta = litere_posibile.index(litera_posibila)
                    factori[index_litera_existenta] += factor
            # print()
            litere_posibile_placuta.append(litere_posibile)
            factori_placuta.append(factori)

        for i in range(0, len(strings_unique[maximum_index])):
            pass
            # print(f'Pentru caracterul de pe indexul {i} luam in considerare: {litere_posibile_placuta[i]} cu 
            # factorii {factori_placuta[i]}') 
        # print()

        string_final = ''
        for index, elem in enumerate(litere_posibile_placuta):
            if len(elem) == 1:
                # print(f'Pentru indexul {index} am ales {elem[0]}')
                string_final = string_final + elem[0]
            else:
                maximum_local = max(factori_placuta[index])
                maximum_local_index = factori_placuta[index].index(maximum_local)
                # print(f'Pentru indexul {index} am ales {elem[maximum_local_index]}')
                string_final = string_final + elem[maximum_local_index]

    string_final = string_final.replace(' ', '')
    string_final = string_final.upper()

    print(f'\nPlacuta finala: {string_final}')
    # print(f'\nPreluata la data de: {date_kept}')
    # print(f'\nOra: {timestamp_kept}')
    # print(f'\nFisierul: {filename_kept}')
    # print('================================')
    return [string_final, date_kept, timestamp_kept, filename_kept]


# --- 11. d) Compute final plates
# --- 11. e) Postprocess CSV files containing detection info

# ----- 12. Send email alerts

smtp_server = "smtp.mail.yahoo.com"
port = 587  # For starttls
sender = 'anprthesis@yahoo.com'
receiver = 'anprthesis@gmail.com'
password = 'wamxfatshsvkanyb'

# Create a secure SSL context
context = ssl.create_default_context()


def send_alert(follower_type, follower_plate):
    """Send mail alerts function.

    Parameters:
        follower_type (str): follower type
        follower_plate (str): follower plate

    Returns: N/A"""

    # Try to log in to server and send email
    server = smtplib.SMTP(smtp_server, port)
    try:
        server.ehlo()
        server.starttls(context=context)  # Secure the connection
        server.ehlo()
        server.login(sender, password)

        email_msg = EmailMessage()

        now_moment = datetime.now()
        date_string = now_moment.strftime('%y-%m-%d')
        time_string = now_moment.strftime('%H-%M-%S')

        msg = f'You are being followed by {follower_plate}, date {date_string}, time {time_string}.\nType of follower: {follower_type}. '

        email_msg.set_content(msg)

        email_msg['Subject'] = f'Dangerous car behind!'
        email_msg['From'] = sender
        email_msg['To'] = receiver

        server.send_message(email_msg)

    except Exception as e:
        print(e)

    finally:
        server.quit()


# ----- 13. Threading, Real Time Detections, Processing of Detections, Alert

# File cleanup
src = 'RealTimeDetections\\NoBlanks'
trg = 'RealTimeDetections\\ForDetections'
deleteProcessed = 'RealTimeDetections\\Processed'
deleteNoBlanks = 'RealTimeDetections\\NoBlanks'

sourceFiles = os.listdir(src)
targetFiles = os.listdir(trg)
deleteProcessedFiles = os.listdir(deleteProcessed)
deleteNoBlanksFiles = os.listdir(deleteNoBlanks)

for fname in targetFiles:
    if isfile(join(trg, fname)) and fname.endswith('.csv'):
        os.remove(os.path.join(trg, fname))

# Copying files used for alerts to their correct directory
for fname in sourceFiles:
    shutil.copy2(os.path.join(src, fname), trg)

for fname in deleteNoBlanksFiles:
    if isfile(join(deleteNoBlanks, fname)) and fname.endswith('.csv'):
        os.remove(os.path.join(deleteNoBlanks, fname))
for fname in deleteProcessedFiles:
    if isfile(join(deleteProcessed, fname)) and fname.endswith('.csv'):
        os.remove(os.path.join(deleteProcessed, fname))


def task1():
    """
    Task 1 for parallel processing. Real Time detections and alerts.
    """

    # Computing a list of previous detections
    csv_for_detections_files = [f_csv for f_csv in listdir('RealTimeDetections\\ForDetections') if
                                isfile(join('RealTimeDetections\\ForDetections', f_csv)) and f_csv.endswith('.csv')]

    previous_detections = []
    for elemCsvForDetections in csv_for_detections_files:

        detections_task = pd.read_csv(os.path.join('RealTimeDetections\\ForDetections', elemCsvForDetections),
                                      header=None,
                                      usecols=[3])
        list_detections = detections_task.values.tolist()
        list_detections_new = [elem[0] for elem in list_detections]
        list_detections_new_new = [elem[2:-2] for elem in list_detections_new]
        for elem in list_detections_new_new:
            previous_detections.append(elem)

    # Real Time Detections from Webcam
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    now_detections = []
    while cap.isOpened():
        ret, frame = cap.read()

        # Preprocessing image
        img_resized_local = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        img_details_enhanced_local = cv2.detailEnhance(img_resized_local, sigma_s=10, sigma_r=0.15)
        img_bilateral_filter_enhanced_local = cv2.bilateralFilter(img_details_enhanced_local, 9, 75, 75)
        img_bilateral_filter_enhanced_gray_local = cv2.cvtColor(img_bilateral_filter_enhanced_local, cv2.COLOR_BGR2GRAY)
        img_thresh_enhanced_local = \
            cv2.threshold(cv2.medianBlur(img_bilateral_filter_enhanced_gray_local, 3), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
                1]
        img_backtorgb_local = cv2.cvtColor(img_thresh_enhanced_local, cv2.COLOR_GRAY2RGB)

        image_np_local = np.array(frame)
        image_np_preprocessed_local = np.array(img_backtorgb_local)

        input_tensor_local = tf.convert_to_tensor(np.expand_dims(image_np_preprocessed_local, 0), dtype=tf.float32)
        detections_task = detect_fn(input_tensor_local)

        num_detections_local = int(detections_task.pop('num_detections'))
        detections_task = {key: value[0, :num_detections_local].numpy()
                           for key, value in detections_task.items()}
        detections_task['num_detections'] = num_detections_local

        # Detection_classes should be ints.
        detections_task['detection_classes'] = detections_task['detection_classes'].astype(np.int64)

        label_id_offset_local = 1
        image_np_with_detections_local = image_np_local.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections_local,
                                                            detections_task['detection_boxes'],
                                                            detections_task[
                                                                'detection_classes'] + label_id_offset_local,
                                                            detections_task['detection_scores'], category_index,
                                                            use_normalized_coordinates=True, max_boxes_to_draw=5,
                                                            min_score_thresh=.8, agnostic_mode=False)

        try:
            text, region = ocr_it(image_np_with_detections_local, detections_task, detection_threshold,
                                  region_threshold)
            save_results(text, region)

            to_check = text[0]
            to_check = to_check.replace(' ', '')
            to_check = to_check.upper()

            # Sending alerts
            if (to_check in previous_detections) or (now_detections.count(to_check) > 1):
                now_detections.append(to_check)
                if (to_check in previous_detections) and (now_detections.count(to_check) > 1):
                    type_of_following = 'detected previously and following you now'
                    print(f'Danger, {type_of_following}.')
                    send_alert(type_of_following, to_check)
                elif to_check in previous_detections:
                    type_of_following = 'detected in a previous moment'
                    print(f'Danger, {type_of_following}.')
                    send_alert(type_of_following, to_check)
                else:
                    type_of_following = 'detected for too long in this moment'
                    print(f'Danger, {type_of_following}.')
                    send_alert(type_of_following, to_check)
            else:
                print('Safe')

        except:
            pass

        cv2.imshow('object detection', cv2.resize(image_np_with_detections_local, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


def task2():
    """
    Task 2 for parallel processing. Cleanup of files, processing past detections.
    """

    csvfiles = [f_csv_ for f_csv_ in listdir('RealTimeDetections') if
                isfile(join('RealTimeDetections', f_csv_)) and f_csv_.endswith('.csv')]
    # First one doesn't conform to the template
    # csvfiles = csvfiles[2:]
    # print(csvfiles)

    for elem_csv_files in csvfiles:
        csv_returns = []
        red_returns = []
        final_plate = ''

        [csv_list_date_return, csv_list_time_return, csv_list_name_return, csv_list_plate_return] = get_data_from_csv(
            elem_csv_files)
        csv_returns.extend([csv_list_date_return, csv_list_time_return, csv_list_name_return, csv_list_plate_return])

        # print(csv_returns[0])
        # print()
        # print(csv_returns[1])
        # print()
        # print(csv_returns[2])
        # print()
        # print(csv_returns[3])
        # print('---------------------------------------')

        [red_list_date_return, red_list_time_return, red_list_file_return, red_list_plate_return] = redundancy_function(
            csv_returns[0], csv_returns[1], csv_returns[2], csv_returns[3])
        red_returns.extend([red_list_date_return, red_list_time_return, red_list_file_return, red_list_plate_return])

        # print(red_returns[0])
        # print()
        # print(red_returns[1])
        # print()
        # print(red_returns[2])
        # print()
        # print(red_returns[3])
        # print('---------------------------------------')

        for i, platesToProc in enumerate(red_returns[3]):
            final_plate = process_plate(platesToProc, red_returns[1][i], red_returns[0][i], red_returns[2][i])

            f_write_csv = open(join('RealTimeDetections\\Processed', f'Processed_{final_plate[1]}.csv'), 'a')

            # create the csv writer
            writer = csv.writer(f_write_csv)

            row = [final_plate[1], final_plate[2], final_plate[3], final_plate[0]]
            # write a row to the csv file
            writer.writerow(row)

            # close the file
            f_write_csv.close()

        csv_processed_files = [f_csv_proc for f_csv_proc in listdir('RealTimeDetections\\Processed') if
                               isfile(join('RealTimeDetections\\Processed', f_csv_proc)) and f_csv_proc.endswith(
                                   '.csv')]

        for elem_input in csv_processed_files:
            find_index = elem_input.find('.csv')
            elem_output = elem_input[:find_index]
            elem_output = elem_output + "_NoBlanks.csv"

            with open(join('RealTimeDetections\\Processed', elem_input), 'r') as inputFile, open(
                    join('RealTimeDetections\\NoBlanks', elem_output), 'w', newline='') as outputFile:
                writer = csv.writer(outputFile)
                for row in csv.reader(inputFile):
                    if any(field.strip() for field in row):
                        writer.writerow(row)


start_time = perf_counter()

# Create two new threads
t1 = Thread(target=task1)
t2 = Thread(target=task2)

# Start the threads
t1.start()
t2.start()

# Wait for the threads to complete
t1.join()
t2.join()

end_time = perf_counter()

print(f'It took {end_time - start_time: 0.2f} second(s) to complete.')

# ----- 14. Freezing the Graph
# ----- 15. Conversion to TFJS
# ----- 16. Conversion to TFLite
