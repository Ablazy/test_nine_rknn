import os
import numpy as np
from crop_image import draw_points_on_image,bytes_to_pil,validate_path
import time
from PIL import Image, ImageDraw
from rknnlite.api import RKNNLite as RKNN
from operators import rknn_infer_dfine
import os
# os.environ["RKNN_LOG_LEVEL"] = "5"

    
def calculate_iou(boxA, boxB):
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])

    intersection_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = float(boxA_area + boxB_area - intersection_area)
    if union_area == 0:
        return 0.0 
    iou = intersection_area / union_area
    return iou

def non_maximum_suppression(detections, iou_threshold=0.35):
    if not detections:
        return []
    detections.sort(key=lambda x: x['score'], reverse=True)

    final_detections = []
    while detections:
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        detections_to_keep = []
        for det in detections:
            iou = calculate_iou(best_detection['box'], det['box'])
            if iou < iou_threshold:
                detections_to_keep.append(det)
        detections = detections_to_keep

    return final_detections

def load_pdl_rknn_model(name='PP-HGNetV2-B4.rknn'):
    global rknn_pdl
    start = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model', name)
    rknn_pdl = RKNN()
    ret = rknn_pdl.load_rknn(model_path)
    if ret != 0:
        print("加载模型失败")
        exit(ret)
    ret = rknn_pdl.init_runtime()
    if ret != 0:
        print("初始化运行时失败")
        exit(ret)
    print(f"加载{name}模型，耗时:{time.time() - start}")

def predict_rknn_dfine(image, draw_result=False):
    # load_dfine_rknn_model()
    rknn_infer_dfine.load_model('model/d-fine-n_op.rknn')
    if isinstance(image, bytes):
        im_pil = bytes_to_pil(image)
    else:
        im_pil = Image.open(image).convert("RGB")
    w, h = im_pil.size
    # 这里的orig_size_np需要是4维的，形状为(1, 1, 1, 2)，因为RKNN模型需要4维输入
    orig_size_np = np.array([[[[w, h]]]], dtype=np.int64)
    im_resized = im_pil.resize((320, 320), Image.Resampling.BILINEAR)
    im_data = np.array(im_resized, dtype=np.float32) / 255
    # 是否转置，测试来看，使用NCWH和NWHC在C API的推理上没有区别，原因不明
    # 在C API中，捕获模型输入的fmt，值为NHWC
    # im_data = im_data.transpose(2, 0, 1)
    im_data = np.expand_dims(im_data, axis=0)
    start = time.time()
    # outputs = rknn_dfine.inference(inputs=[im_data, orig_size_np])[0]
    outputs = rknn_infer_dfine.infer_rknn(im_data, orig_size_np)
    print(f"rknn推理耗时: {time.time() - start}")
    # rknn_dfine.release()
    
    labels = outputs[0]
    boxes = outputs[1]
    scores = outputs[2]
    # print(labels)
    # print(scores)
    # print(f"labels:{labels.shape}, boxes:{boxes.shape}, scores:{scores.shape}")

    colors = ["red", "blue", "green", "yellow", "white", "purple", "orange"]
    mask = scores > 0.4
    filtered_labels = labels[mask]
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]

    rebuild_color = {}
    unique_labels = list(set(filtered_labels))
    for i, l_val in enumerate(unique_labels):
        class_id = int(l_val)
        if class_id not in rebuild_color:
            rebuild_color[class_id] = colors[i % len(colors)]
    
    result = {k: [] for k in unique_labels}

    for i, box in enumerate(filtered_boxes):
        if box[2]>160 and box[3] < 45:
            continue
        label_val = filtered_labels[i]
        class_id = int(label_val)
        color = rebuild_color[class_id]
        score = filtered_scores[i]
        
        result[class_id].append({
            'box': box,
            'label_val': label_val,
            'score': score
        })

    keep_result = {}
    result_points = []
    for class_id in result:
        tp = non_maximum_suppression(result[class_id],0.01)
        if len(tp) < 2:
            continue
        point = tp[0]["score"]+tp[1]["score"]
        if point < 0.85:
            continue
        keep_result.update({class_id:tp[0:2]})
        result_points.append({"id":class_id,"point":point})
    result_points.sort(key=lambda item: item['point'], reverse=True)
    if len(keep_result) > 3:
        tp = {}
        for one in result_points[0:3]:
            tp.update({one['id']:keep_result[one['id']]})
        keep_result = tp
    for class_id in keep_result:
        keep_result[class_id].sort(key=lambda item: item['box'][3], reverse=True)
    sorted_result = {}
    sorted_class_ids = sorted(keep_result.keys(), key=lambda cid: keep_result[cid][0]['box'][0])
    for class_id in sorted_class_ids:
        sorted_result[class_id] = keep_result[class_id]
    points = []

    if draw_result:
        draw = ImageDraw.Draw(im_pil)
    for c1,class_id in enumerate(sorted_result):
        items = sorted_result[class_id]
        last_item = items[-1]
        center_x = (last_item['box'][0] + last_item['box'][2]) / 2
        center_y = (last_item['box'][1] + last_item['box'][3]) / 2
        text_position_center = (center_x , center_y)
        points.append(text_position_center)
        if draw_result:
            color = rebuild_color[class_id]
            draw.point((center_x, center_y), fill=color)
            text_center = f"{c1}"
            draw.text(text_position_center, text_center, fill=color)
            for c2,item in enumerate(items):
                box = item['box']
                score = item['score']
                
                draw.rectangle(list(box), outline=color, width=1)
                text = f"{class_id}_{c1}-{c2}: {score:.2f}"
                text_position = (box[0] + 2, box[1] - 12 if box[1] > 12 else box[1] + 2)
                draw.text(text_position, text, fill=color)
    if draw_result:
       save_path = os.path.join(validate_path,"rknn_result.jpg")
       im_pil.save(save_path)
       print(f"图片可视化结果保存在{save_path}")
    print(f"图片顺序的中心点{points}")
    return points

def predict_rknn_pdl(images_path):
    load_pdl_rknn_model()
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
    def data_transforms(path):
        # 打开图片
        img = Image.open(path)
        # 调整图片大小为232x224（假设最短边长度调整为232像素）
        if img.width < img.height:
            new_size = (232, int(232 * img.height / img.width))
        else:
            new_size = (int(232 * img.width / img.height), 232)
        resized_img = img.resize(new_size, Image.BICUBIC)
        # 裁剪图片为224x224
        cropped_img = resized_img.crop((0, 0, 224, 224))
        # # 将图像转换为NumPy数组并进行归一化处理
        # 归一化处理在模型转换是已经完成了
        img_array = np.array(cropped_img).astype(np.float32)
        return img_array
    images = []
    for pic in sorted(os.listdir(images_path)):
        if "cropped" not in pic:
            continue
        image_path = os.path.join(images_path,pic)
        images.append(data_transforms(image_path))
    if len(images) == 0:
        raise FileNotFoundError(f"先使用切图代码切图至{image_path}再推理,图片命名如cropped_9.jpg,从0到9共十个,最后一个是检测目标")
    start = time.time()
    images = np.array(images)
    outputs = rknn_pdl.inference(inputs=[images], data_format='nhwc')[0]
    rknn_pdl.release()
    result = [np.argmax(one) for one in outputs]
    target = result[-1]
    answer = [coordinates[index] for index in range(9) if result[index] == target]
    if len(answer) == 0:
        all_sort =[np.argsort(one) for one in outputs]
        answer = [coordinates[index] for index in range(9) if all_sort[index][1] == target]
    print(f"识别完成{answer}，耗时: {time.time() - start}")
    with open(os.path.join(images_path,"nine.jpg"),'rb') as f:
        bg_image = f.read()
    draw_points_on_image(bg_image, answer)
    return answer

if __name__ == "__main__":
    # 使用PP-HGNetV2-B4.rknn
    #predict_onnx_pdl(r'img_saved\img_fail\7fe559a85bac4c03bc6ea7b2e85325bf')
    # predict_rknn_pdl(r'img_2_val')
    # predict_rknn_pdl(r'img_saved/img_fail/933762ddd6054c13a4d90740180afe51')
    # 使用d-fine-n_op.rknn
    # load_dfine_model()
    img_path = r"test_data/test_icon.jpg"
    predict_rknn_dfine(img_path,True)
