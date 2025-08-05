from rknn.api import RKNN
from operators.register_gridsample_op import cstGridSample
from PIL import Image
import numpy as np

ONNX_MODEL = 'model/d-fine-n_modified_clip.onnx'
RKNN_MODEL = 'model/d-fine-n.rknn'

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(
        # mean_values=[[123.675, 116.28, 103.53], [0]],
        # std_values=[[58.395, 57.12, 57.375], [1]],
        mean_values=[[0, 0, 0], [0]],
        std_values=[[1, 1, 1], [1]],
        target_platform='rk3588',
        # dynamic_input=[[[1,3,320,320], [1,2]]]
        )
    print('done')

    # Register custom op
    print('--> Register cstGridSample op')
    ret = rknn.reg_custom_op(cstGridSample())
    if ret != 0:
        print('Register cstGridSample op failed!')
        exit(ret)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(
        model=ONNX_MODEL,
        inputs=['images', 'orig_target_sizes'],
        input_size_list=[[1, 3, 320, 320],[1, 1, 1, 2]]
    )
    # ret = rknn.load_rknn("model/d-fine-n_op.rknn")
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    ### stimulate the inference process ###

    # # Set inputs
    # image = "test.jpg"
    # im_pil = Image.open(image).convert("RGB")
    
    # w, h = im_pil.size
    # orig_size_np = np.array([[[[w, h]]]], dtype=np.int64)
    # print(f'Original image size: {w}x{h}')
    # np.save('orig_size.npy', orig_size_np)
    # im_resized = im_pil.resize((320, 320), Image.Resampling.BILINEAR)
    # im_data = np.array(im_resized, dtype=np.float32)/255
    # im_data = im_data.transpose((2, 0, 1))  # Convert to CHW format
    # im_data = np.expand_dims(im_data, axis=0)
    # np.save('im_data.npy', im_data)
    # # Init runtime environment
    # print('--> Init runtime environment')
    # # ret = rknn.init_runtime(target='rk3588', device_id='192.168.1.6:5555')
    # ret = rknn.init_runtime()
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('Init runtime environment done')

    # ret = rknn.accuracy_analysis(inputs=["im_data.npy", "orig_size.npy"],)
    # # Inference
    # print('--> Running model')
    # outputs = rknn.inference(inputs=[im_data, orig_size_np], data_format=['nchw', 'nchw'])
    # print(outputs)
    # # x = outputs[0]
    # # output = np.exp(x)/np.sum(np.exp(x))
    # # outputs = [output]
    # # show_outputs(outputs)
    # print('done')

    rknn.release()