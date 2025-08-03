import numpy as np
from rknn.api.custom_op import get_node_attr

class cstGridSample:
    op_type = 'cstGridSample'
    
    def shape_infer(self, node, in_shapes, in_dtypes):
        # 输入形状: [input_shape, grid_shape]
        # input_shape: [N, C, H, W]
        # grid_shape: [N, H_out, W_out, 2]
        # 输出形状: [N, C, H_out, W_out]
        input_shape = in_shapes[0]
        grid_shape = in_shapes[1]
        
        N, C = input_shape[0], input_shape[1]
        H_out, W_out = grid_shape[1], grid_shape[2]
        
        out_shapes = [[N, C, H_out, W_out]]
        out_dtypes = [in_dtypes[0]]  # 输出数据类型与输入相同
        
        return out_shapes, out_dtypes
    
    def compute(self, node, inputs):
        input_tensor = inputs[0]  # [N, C, H, W]
        grid = inputs[1]          # [N, H_out, W_out, 2]
        
        # 获取属性
        mode = get_node_attr(node, 'mode')  # 插值模式
        padding_mode = get_node_attr(node, 'padding_mode')  # 填充模式
        align_corners = get_node_attr(node, 'align_corners')  # 是否对齐角点
        
        # 执行grid_sample计算
        output = self._grid_sample(input_tensor, grid, mode, padding_mode, align_corners)
        return [output]
    
    def _grid_sample(self, input_tensor, grid, mode, padding_mode, align_corners):
        N, C, H, W = input_tensor.shape
        _, H_out, W_out, _ = grid.shape
        
        # 根据align_corners选择坐标变换方式
        if align_corners:
            # 对齐角点：[-1, 1] -> [0, W-1] 和 [0, H-1]
            grid_x = (grid[..., 0] + 1.0) * (W - 1) / 2.0
            grid_y = (grid[..., 1] + 1.0) * (H - 1) / 2.0
        else:
            # 不对齐角点：[-1, 1] -> [-0.5, W-0.5] 和 [-0.5, H-0.5]
            grid_x = ((grid[..., 0] + 1.0) * W - 1.0) / 2.0
            grid_y = ((grid[..., 1] + 1.0) * H - 1.0) / 2.0
        
        if mode == 'bilinear':
            return self._bilinear_sample(input_tensor, grid_x, grid_y, padding_mode)
        else:
            # nearest模式
            return self._nearest_sample(input_tensor, grid_x, grid_y, padding_mode)
    
    def _bilinear_sample(self, input_tensor, grid_x, grid_y, padding_mode):
        N, C, H, W = input_tensor.shape
        H_out, W_out = grid_x.shape[1], grid_x.shape[2]
        
        # 获取四个相邻像素的坐标
        x0 = np.floor(grid_x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(grid_y).astype(np.int32)
        y1 = y0 + 1
        
        # 计算插值权重
        wx = grid_x - x0.astype(np.float32)
        wy = grid_y - y0.astype(np.float32)
        
        # 边界检查
        valid_x0 = (x0 >= 0) & (x0 < W)
        valid_x1 = (x1 >= 0) & (x1 < W)
        valid_y0 = (y0 >= 0) & (y0 < H)
        valid_y1 = (y1 >= 0) & (y1 < H)
        
        # 确保索引在有效范围内（用于数组索引）
        x0 = np.clip(x0, 0, W - 1)
        x1 = np.clip(x1, 0, W - 1)
        y0 = np.clip(y0, 0, H - 1)
        y1 = np.clip(y1, 0, H - 1)
        
        # 初始化输出数组
        output = np.zeros((N, C, H_out, W_out), dtype=input_tensor.dtype)
        
        # 批量处理每个样本
        for n in range(N):
            for c in range(C):
                # 获取四个角点的值
                Q00 = input_tensor[n, c, y0[n], x0[n]]  # 左上
                Q10 = input_tensor[n, c, y0[n], x1[n]]  # 右上
                Q01 = input_tensor[n, c, y1[n], x0[n]]  # 左下
                Q11 = input_tensor[n, c, y1[n], x1[n]]  # 右下
                
                # 根据填充模式处理边界外的值
                if padding_mode == 'zeros':
                    Q00 = np.where(valid_x0[n] & valid_y0[n], Q00, 0)
                    Q10 = np.where(valid_x1[n] & valid_y0[n], Q10, 0)
                    Q01 = np.where(valid_x0[n] & valid_y1[n], Q01, 0)
                    Q11 = np.where(valid_x1[n] & valid_y1[n], Q11, 0)
                elif padding_mode == 'border':
                    # 边界填充模式（使用边界值）
                    x0_clip = np.clip(x0[n], 0, W - 1)
                    x1_clip = np.clip(x1[n], 0, W - 1)
                    y0_clip = np.clip(y0[n], 0, H - 1)
                    y1_clip = np.clip(y1[n], 0, H - 1)
                    Q00 = input_tensor[n, c, y0_clip, x0_clip]
                    Q10 = input_tensor[n, c, y0_clip, x1_clip]
                    Q01 = input_tensor[n, c, y1_clip, x0_clip]
                    Q11 = input_tensor[n, c, y1_clip, x1_clip]
                
                # 双线性插值计算
                # 先在x方向插值
                Q0 = Q00 * (1 - wx[n]) + Q10 * wx[n]
                Q1 = Q01 * (1 - wx[n]) + Q11 * wx[n]
                
                # 再在y方向插值
                output[n, c, :, :] = Q0 * (1 - wy[n]) + Q1 * wy[n]
        
        return output
    
    def _nearest_sample(self, input_tensor, grid_x, grid_y, padding_mode):
        N, C, H, W = input_tensor.shape
        H_out, W_out = grid_x.shape[1], grid_x.shape[2]
        
        # 获取最近邻像素的坐标
        x_nearest = np.round(grid_x).astype(np.int32)
        y_nearest = np.round(grid_y).astype(np.int32)
        
        # 边界检查
        valid_x = (x_nearest >= 0) & (x_nearest < W)
        valid_y = (y_nearest >= 0) & (y_nearest < H)
        
        # 确保索引在有效范围内
        x_nearest = np.clip(x_nearest, 0, W - 1)
        y_nearest = np.clip(y_nearest, 0, H - 1)
        
        # 初始化输出数组
        output = np.zeros((N, C, H_out, W_out), dtype=input_tensor.dtype)
        
        # 批量处理
        for n in range(N):
            for c in range(C):
                # 获取最近邻点的值
                values = input_tensor[n, c, y_nearest[n], x_nearest[n]]
                
                # 根据填充模式处理边界外的值
                if padding_mode == 'zeros':
                    values = np.where(valid_x[n] & valid_y[n], values, 0)
                elif padding_mode == 'border':
                    # 边界填充模式直接使用裁剪后的索引
                    x_clip = np.clip(x_nearest[n], 0, W - 1)
                    y_clip = np.clip(y_nearest[n], 0, H - 1)
                    values = input_tensor[n, c, y_clip, x_clip]
                
                output[n, c, :, :] = values
        
        return output