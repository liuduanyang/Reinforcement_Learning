import numpy as np

'''
参考：https://towardsdatascience.com/reinforcement-learning-tile-coding-implementation-7974b600762b
'''

def create_tiling(feat_range, bins, offset):   # 一维tiling
    '''
    创建一维度的tiling空间代表一个特征  \n
    feature_range：特征范围  \n
    bins：分位点数 特征内的元素个数  \n
    offset：特征的偏置
    '''


    '''
    linespace 会从参数1-参数2的范围按照均匀的间隔划分为参数3个数 (间隔数为参数3-1)
    '''
    return np.linspace(feat_range[0], feat_range[1], bins+1)[1:-1] + offset

feat_range = [0, 1.0]
bins = 10
offset = 0.2

tiling_spec = create_tiling(feat_range, bins, offset)
print(tiling_spec)


def create_tilings(feat_ranges, number_tilings, bins, offsets):   # 多维tiling
    '''
    feature_ranges：多维特征的范围  如 [[-1,1], [2,5]]
    number_tilings：tiling的个数
    bins：每一个tiling的尺寸
    offset：每一个tiling的偏置
    '''
    tilings = []
    # 对每一tiling
    for tile_i in range(number_tilings):
        tiling_bin = bins[tile_i]
        tiling_offset = offsets[tile_i]

        tiling = []
        # 对于每一特征维度
        for feat_i in range(len(feature_ranges)):
            feat_range = feature_ranges[feat_i]
            feat_tiling = create_tiling(feat_range, tiling_bin[feat_i], tiling_offset[feat_i])
            tiling.append(feat_tiling)
        
        tilings.append(tiling)
    
    return np.array(tilings)

feature_ranges = [[-1,1], [2,5]]
number_tilings = 3
bins = [[10,10], [10,10], [10,10]]
offsets = [[0,0], [0.2,1], [0.4,1.5]]

tilings = create_tilings(feature_ranges, number_tilings, bins, offsets)
print(tilings.shape)


def get_tile_coding(feature, tilings):
    '''
    feature：多维度下需要编码的特征值
    tilings：由create_tilings生成
    '''
    num_dims = len(feature)
    feat_codings = []
    for tiling in tilings:
        feat_coding = []
        for i in range(num_dims):
            feat_i = feature[i]
            tiling_i = tiling[i]
            coding_i = np.digitize(feat_i, tiling_i)   # 返回feat_i在tiling_i的第几个区间内
            feat_coding.append(coding_i)
        feat_codings.append(feat_coding)
    
    return np.array(feat_codings)

feature = [0.1, 2.5]

coding = get_tile_coding(feature, tilings)
print(coding)