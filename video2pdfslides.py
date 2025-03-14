import os
import time
import cv2
import imutils
import shutil
import img2pdf
import glob
import argparse

############# Define constants

OUTPUT_SLIDES_DIR = f"/home/marcochsner/Documents/trunk/_3-CUBCS/DataMining/data-mining-methods/Week\ 01"

FRAME_RATE = 1                   # no.of frames per second that needs to be processed, fewer the count faster the speed
WARMUP = FRAME_RATE              # initial number of frames to be skipped
FGBG_HISTORY = FRAME_RATE * 15   # no.of frames in background object
VAR_THRESHOLD = 16               # Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
DETECT_SHADOWS = False            # If true, the algorithm will detect shadows and mark them.
MIN_PERCENT = 0.1                # min % of diff between foreground and background to detect if motion has stopped
MAX_PERCENT = 3                  # max % of diff between foreground and background to detect if frame is still in motion


def get_frames(video_path):
    '''A fucntion to return the frames from a video located at video_path
    this function skips frames as defined in FRAME_RATE'''
    
    
    # open a pointer to the video file initialize the width and height of the frame
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'unable to open file {video_path}')


    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_time = 0
    frame_count = 0
    print("total_frames: ", total_frames)
    print("FRAME_RATE", FRAME_RATE)

    # loop over the frames of the video
    while True:
        # grab a frame from the video

        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)    # move frame to a timestamp
        frame_time += 1/FRAME_RATE # OVERRIDE frame_rate by 5s

        (_, frame) = vs.read()
        # if the frame is None, then we have reached the end of the video file
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()
 


def detect_unique_screenshots(video_path, output_folder_screenshot_path, md_open_file):
    """
    # Initialize fgbg a Background object with Parameters
    # history = The number of frames history that effects the background subtractor
    # varThreshold = Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.
    # detectShadows = If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.
    """
    fgbg = cv2.createBackgroundSubtractorMOG2(history=FGBG_HISTORY, varThreshold=VAR_THRESHOLD,detectShadows=DETECT_SHADOWS)

    
    captured = False
    start_time = time.time()
    (W, H) = (None, None)

    screenshoots_count = 0
    for frame_count, frame_time, frame in get_frames(video_path):
        orig = frame.copy() # clone the original frame (so we can save it later), 
        frame = imutils.resize(frame, width=600) # resize the frame
        mask = fgbg.apply(frame) # apply the background subtractor

        # apply a series of erosions and dilations to eliminate noise
#            eroded_mask = cv2.erode(mask, None, iterations=2)
#            mask = cv2.dilate(mask, None, iterations=2)

        # if the width and height are empty, grab the spatial dimensions
        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # compute the percentage of the mask that is "foreground"
        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        # if p_diff less than N% then motion has stopped, thus capture the frame

        if p_diff < MIN_PERCENT and not captured and frame_count > WARMUP:
            captured = True
            filename = f"{screenshoots_count:03}_{round(frame_time/60, 2)}.png"

            path = os.path.join(output_folder_screenshot_path, filename)
            print("saving {}".format(path))
            cv2.imwrite(path, orig)

            image_ref = path.rsplit("trunk")[-1]
            image_ref = image_ref.replace(" ", "%20")

            # ![](_3-CUBCS/DataMining/CSCA5502-DataMiningPipeline/4.2-CorrelationAnalysis/000_0.03.png)
            new_line = "![](" + image_ref + ")\n"
            md_open_file.write(new_line)
            screenshoots_count += 1

        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model
        elif captured and p_diff >= MAX_PERCENT:
            captured = False
    print(f'{screenshoots_count} screenshots Captured!')
    print(f'Time taken {time.time()-start_time}s')
    return 


def initialize_output_folder(video_path):
    '''Clean the output folder if already exists'''
    output_folder_screenshot_path = f"{OUTPUT_SLIDES_DIR}/{video_path.rsplit('/')[-1].split('.')[0]}"

    if os.path.exists(output_folder_screenshot_path):
        shutil.rmtree(output_folder_screenshot_path)

    os.makedirs(output_folder_screenshot_path, exist_ok=True)
    print('initialized output folder', output_folder_screenshot_path)
    return output_folder_screenshot_path


def convert_screenshots_to_pdf(output_folder_screenshot_path):
    output_pdf_path = f"{OUTPUT_SLIDES_DIR}/{video_path.rsplit('/')[-1].split('.')[0]}" + '.pdf'
    print('output_folder_screenshot_path', output_folder_screenshot_path)
    print('output_pdf_path', output_pdf_path)
    print('converting images to pdf..')
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(sorted(glob.glob(f"{output_folder_screenshot_path}/*.png"))))
    print('Pdf Created!')
    print('pdf saved at', output_pdf_path)


if __name__ == "__main__":
    video_paths = [
        # "/home/marcochsner/Documents/trunk/_3-CUBCS/DataMining/data-mining-methods/Week 01/07 - Data Mining Technique View/01 - Lecture video (720p).mp4",
    ]
    
    curr_path = "/home/marcochsner/Documents/trunk/_3-CUBCS/DataMining/data-mining-methods/Week 01/"
    dir_list = os.listdir(curr_path)
    for d in dir_list:
        if os.path.isfile(os.path.join(curr_path, d)):
            continue
        file_list = os.listdir(os.path.join(curr_path, d))
        for f in file_list:
            if not os.path.isdir(os.path.join(curr_path, d)):
                continue
            if ".mp4" in f:
                video_paths.append(os.path.join(curr_path, d, f))
    #     output_folder_screenshot_path = initialize_output_folder(video_path)
    #     detect_unique_screenshots(video_path, output_folder_screenshot_path)
    #     convert_screenshots_to_pdf(output_folder_screenshot_path)
    COURSERA_DL_FORMAT = False
    if COURSERA_DL_FORMAT:
        ...
    elif len(video_paths) > 0:
        for video_path in video_paths:
            output_path = "/home/marcochsner/Documents/trunk/_3-CUBCS/Networks/Networks1/Wk1/"
            choice = 'y'
            output_folder_screenshot_path = initialize_output_folder(video_path)
            print('video_path', video_path)
            # output_folder_screenshot_path = initialize_output_folder(video_path)
            video_path_array = video_path.rsplit("/")
            coursera_dl_format = True
            base_name_no_extension = ("/".join(video_path_array[:-1]) + "/" + ".".join(video_path_array[-1].rsplit(".")[:-1])) \
                if not coursera_dl_format \
                else ("/".join(video_path_array[:-2]) + "/" + (video_path_array[-2]))

            image_output_path = base_name_no_extension + "/"
            markdown_path = base_name_no_extension + ".md"

            os.makedirs(image_output_path, exist_ok=True)
            md_open_file = open(markdown_path, "w")  # Create if DNE

            detect_unique_screenshots(video_path, image_output_path, md_open_file)
            md_open_file.close()

            print('Please Manually verify screenshots and delete duplicates')
            while True:
                # choice = input("Press y to continue and n to terminate")
                choice = choice.lower().strip()
                if choice in ['y', 'n']:
                    break
                else:
                    print('please enter a valid choice')

            # if choice == 'y':
                # convert_screenshots_to_pdf(output_folder_screenshot_path)
    else:
        parser = argparse.ArgumentParser(prog="ProgramName", description="What the program does")
        parser.add_argument("video_path", help="path of video to be converted to pdf slides", type=str)
        parser.add_argument("-o", "--output", help="output folder relative or not", type=str)
        # parser.add_argument("-m", "--markdown-file", help="output folder relative or not", type=str)
        # parser.add_argument("-g", "--gen-markdown-file", type=bool)

        args = parser.parse_args()
        video_path = args.video_path
        output_path = args.output
        # markdown_file = args.markdown_file
        # gen_markdown_file = args.gen_markdown_file
        print('video_path', video_path)
        # output_folder_screenshot_path = initialize_output_folder(video_path)
        video_path_array = video_path.rsplit("/")
        base_name_no_extension = "/".join(video_path_array[:-1]) + "/" + ".".join(video_path_array[-1].rsplit(".")[:-1])
        image_output_path = base_name_no_extension + "/"
        markdown_path = base_name_no_extension + ".md"

        os.makedirs(image_output_path, exist_ok=True)
        md_open_file = open(markdown_path, "w") # Create if DNE

        detect_unique_screenshots(video_path, image_output_path, md_open_file)
        md_open_file.close()

        print('Please Manually verify screenshots and delete duplicates')
        while True:
            choice = input("Press y to continue and n to terminate")
            choice = choice.lower().strip()
            if choice in ['y', 'n']:
                break
            else:
                print('please enter a valid choice')

        # if choice == 'y':
        #     convert_screenshots_to_pdf(output_folder_screenshot_path)


