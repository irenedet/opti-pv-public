import argparse
import sys
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import supervisely_lib as sly
import yaml
import time
import torchvision

from PIL import Image
from torchvision import transforms

FILE = Path('__file__').absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, xywh2xyxy
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.functions import to_numpy, construct_model_meta, prepare_model, sliding_window_approach, removeduplicate, \
    infer_onnx_model, non_max_suppression_sly, infer_torch_model


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.60,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix = False, Path(w).suffix.lower()
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in ['.pt', '.onnx', '.tflite', '.pb', ''])
    stride, names = 64, [f'class{i}' for i in range(1000)]
    if pt:
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        names = model.module.names if hasattr(model, 'module') else model.names
        if half:
            model.half()
        if classify:
            modelc = load_classifier(name='resnet50', n=2)
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))
            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            int8 = input_details[0]['dtype'] == np.uint8
    imgsz = check_img_size(imgsz, s=stride)
    ascii = is_ascii(names)

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    xyxy_list = []
    for path, img, im0s, vid_cap in dataset:
        print(path[0])
        if onnx:
            img = img.astype('float32')
        else:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
        img = img / 255.0
        if len(img.shape) == 3:
            img = img[None]

        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
        else:
            imn = img.permute(0, 2, 3, 1).cpu().numpy()
            if pb:
                pred = frozen_func(x=tf.constant(imn)).numpy()
            elif saved_model:
                pred = model(imn, training=False).numpy()
            elif tflite:
                if int8:
                    scale, zero_point = input_details[0]['quantization']
                    imn = (imn / scale + zero_point).astype(np.uint8)
                interpreter.set_tensor(input_details[0]['index'], imn)
                interpreter.invoke()
                pred = interpreter.get_tensor(output_details[0]['index'])
                if int8:
                    scale, zero_point = output_details[0]['quantization']
                    pred = (pred.astype(np.float32) - zero_point) * scale
            pred[..., 0] *= imgsz[1]
            pred[..., 1] *= imgsz[0]
            pred[..., 2] *= imgsz[1]
            pred[..., 3] *= imgsz[0]
            pred = torch.tensor(pred)

        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    x1, x2, y1, y2 = to_numpy(x1), to_numpy(x2), to_numpy(y1), to_numpy(y2)
                    xyxy_list.append([[x1, y1, x2, y2], to_numpy(conf), to_numpy(cls)])
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n' +
                                    str(int(x1)) + ' ' + str(int(y1)) + ' ' + str(int(x2)) + ' ' + str(int(y2)))

                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            print(f'{s}Done. ({t2 - t1:.3f}s)')

            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    if len(det):
                        cv2.imwrite(save_path, im0)
                    else:
                        continue
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)

    print(f'Done. ({time.time() - t0:.3f}s)')
    return xyxy_list, save_dir, save_crop, source


def infer_model(model_, image, simple_inference=True, **kwargs):
    if simple_inference:
        infer_fn = infer_onnx_model if isinstance(model_, tuple) else infer_torch_model
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if image.max() > 1:
            image = image / 255
        height, width = kwargs['input_image_size']
        if image.shape[2] > height or image.shape[3] > width:
            image = image[..., :height, :width]
        model_inference = infer_fn(model_, image)
        output = non_max_suppression_sly(model_inference,
                                         conf_thres=kwargs['conf_threshold'],
                                         iou_thres=kwargs['iou_threshold'],
                                         agnostic=kwargs['agnostic'])
    else:
        output = sliding_window_approach(model_, image, **kwargs)
    return output


def visualize_dets(img0, output, save_path, meta, **kwargs):
    labels = []
    model_ = kwargs['model']
    names = model_.module.names if hasattr(model_, 'module') else model_.names
    for i, det in enumerate(output):
        if det is not None and len(det):
            for *xyxy, conf, cls in reversed(det):
                left, top, right, bottom = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                rect = sly.Rectangle(top, left, bottom, right)
                obj_class = meta.get_obj_class(names[int(cls)])
                tag = sly.Tag(meta.get_tag_meta("confidence"), round(float(conf), 4))
                label = sly.Label(rect, obj_class, sly.TagCollection([tag]))
                labels.append(label)

    width, height = img0.size
    ann = sly.Annotation(img_size=(height, width), labels=labels)

    vis = np.copy(img0)
    ann.draw_contour(vis, thickness=3)
    sly.image.write(os.path.join(save_path, 'inf_' + kwargs['filename']), vis)

    ann_json = ann.to_json()
    app = [list(removeduplicate(i['points']['exterior'][0])) for i in ann_json['objects']]
    lbl = len(list(removeduplicate(app)))
    print(str(lbl) + ' defects detected in image: ' + kwargs['filename'])
    return vis


def run_stage2(path_to_images, weights2, output_dir='resultats_inference/'):
    device = select_device(device='cpu')
    kwargs = dict(device=device)
    model, kwargs = prepare_model(weights2, **kwargs)
    meta = construct_model_meta(model, "confidence")
    kwargs['meta'] = meta
    os.makedirs(output_dir, exist_ok=True)

    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(path_to_images) if f.lower().endswith(image_exts)]
    for count, filename in enumerate(files):
        print(f"Processing image {count + 1}/{len(files)}")
        path_to_image = os.path.join(path_to_images, filename)
        kwargs['filename'] = filename

        big_image = Image.open(path_to_image)
        tensor = transforms.PILToTensor()(big_image)
        image = transforms.ToPILImage()(tensor)

        try:
            H, W = model.img_size
            kwargs['input_image_size'] = [H, W]
        except AttributeError:
            with open(kwargs['configs_path'], 'r') as yaml_file:
                cfgs = yaml.safe_load(yaml_file)
            kwargs['input_image_size'] = cfgs['img_size']

        kwargs.update({
            'conf_threshold': 0.25,
            'iou_threshold': 0.45,
            'agnostic': False,
            'sliding_window_step': [320, 320],
            'model': model,
        })

        rez = infer_model(model, tensor, simple_inference=False, **kwargs)
        visualize_dets(img0=image, output=rez, save_path=output_dir, meta=meta, **kwargs)
        print("Done")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='stage 1 model path(s)')
    parser.add_argument('--weights2', type=str, default='models/defect_detector.pt', help='stage 2 model path')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    boxlist, save_dir, save_crop, source = run(**{k: v for k, v in vars(opt).items() if k != 'weights2'})

    if save_crop:
        # use the crops from stage 1 as input to stage 2
        crop_class_dir = str(save_dir) + '/crops/'
        class_dirs = [d for d in os.listdir(crop_class_dir) if os.path.isdir(os.path.join(crop_class_dir, d))]
        path_to_stage2_images = os.path.join(crop_class_dir, class_dirs[0]) if class_dirs else crop_class_dir
    else:
        path_to_stage2_images = source

    run_stage2(path_to_stage2_images, weights2=opt.weights2)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
