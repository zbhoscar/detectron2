# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="../configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )

    ####### For video frames folder use ###############
    parser.add_argument(
        "--input_dataset_dir",
        default="just_not_be_NONE_in_purposes",
        help='For multi-layer-saving if using ./input_imagelist.txt',
    )
    parser.add_argument(
        "--output_txts_dir",
        default="/home/zbh/Absolute/HDD/ucfcrime_test_maskrcnn_frames_txts",
    )
    parser.add_argument(
        "--output_images_dir",
        default="/home/zbh/Absolute/HDD/ucfcrime_test_maskrcnn_frames_images",
    )
    #######################################################

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            ######  FOR ./input_imagelist.txt ######
            # glob.glob(os.path.expanduser(['/home/zbh/Absolute/HDD/ucfcrime_test_frames.txt'][0]))
            if len(args.input) == 1 and args.input[0][-4:] == '.txt':
                with open(args.input[0], 'r') as f:
                    contents = f.readlines()
                args.input = [i.strip() for i in contents]
            ##########################################
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output or args.output_txts_dir or args.output_images_dir:
                # if os.path.isdir(args.output):
                if args.output:
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                # else:
                #     assert len(args.input) == 1, "Please specify a directory with args.output"
                #     out_filename = args.output
                # visualized_output.save(out_filename)
                ###########################################################
                    visualized_output.save(out_filename)
                else:
                    assert args.input_dataset_dir in path, "--input_dataset_dir {} not matches {}".format(args.input_dataset_dir, path)
                    if args.output_txts_dir:
                        output_txt_filename = path.replace(args.input_dataset_dir, args.output_txts_dir)
                        output_txt_filename = output_txt_filename.replace(os.path.splitext(output_txt_filename)[1], '.txt')
                        os.makedirs(os.path.dirname(output_txt_filename), exist_ok=True)

                        boxes = predictions['instances']._fields['pred_boxes'].tensor.cpu().numpy().tolist()
                        scores = predictions['instances']._fields['scores'].cpu().numpy().tolist()
                        classes = predictions['instances']._fields['pred_classes'].cpu().numpy().tolist()

                        with open(output_txt_filename, 'a') as f:
                            for i in range(len(classes)):
                                list_all = boxes[i] + [classes[i]] + [scores[i]]
                                line_string = ' '.join(["{:.4f}".format(elem) for elem in list_all])+'\n'
                                f.writelines(line_string)

                    if args.output_images_dir:
                        output_image_filename = path.replace(args.input_dataset_dir, args.output_images_dir)
                        os.makedirs(os.path.dirname(output_image_filename), exist_ok=True)
                        visualized_output.save(output_image_filename)
                #############################################################
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
