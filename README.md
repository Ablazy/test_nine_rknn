# test_nine_rknn

原项目来自：https://github.com/luguoyixiazi/test_nine
验证码识别，用于极验的九宫格验证码（例如米游社论坛签到验证码）。

部署在具有NPU的瑞芯微（Rockchip）SoC的开发板上，并使用NPU推理模型。

目前实现了PP-HGNetV2-B4模型的NPU推理，可用于九宫格验证。

点选验证使用的d-fine-n模型，由于rknn不支持GridSample算子，无法直接移植。目前使用C API构建了自定义的cstGridSample算子，但遇到不明错位，没有报错信息，暂缓。
