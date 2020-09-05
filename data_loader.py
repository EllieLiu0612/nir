#coding=utf-8
import os
import csv
import random
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft,ifft
from PyEMD import EMD
def aa(signal):
    emd = EMD()
    IMFs = emd(signal)
    return IMFs
def get_fft(data):
    #对强度信号进行傅里叶变换
    s=fft(data)
    #为了滤除高频噪声信号，采用截断函数可以做到这一点
    #根据傅里叶变换的结果设置合适的截断函数
    m=len(s)
    n=50
    cutfun=np.ones([m,1])
    cutfun[20:m-20]=0
    ss =s
    ss[n:m-n]=0 #对傅里叶变换信号做截断
    f=ifft(ss) #逆傅里叶变换
    real_f=np.real(f)
    return real_f

def apply_norm(inp_ys):
    cys = np.array(inp_ys)
    rys = []
    for i in range(cys.shape[-1]):
        ys = cys[:, i]
        maxv, minv = max(ys), min(ys)
        nys = []
        for y in ys:
            nys.append((y - minv) / (maxv - minv))
        rys.append(nys)
    return list(np.array(rys).T)

def _load_csv(csv_path):
    xs, ys = [], []
    with open(csv_path, 'r', encoding='utf8') as fo:
        reader = csv.reader(fo)
        lid = 0
        for row in reader:
            if lid < 35 and (row[0] == '' or row[0][0] != '9'):
                lid += 1
                continue
            else:
                xs.append(float(row[0]))
            ys.append(float(row[1]))
            lid += 1
    return xs, ys

def _load_org(dirs, feat_mask=[1,2]):
    # feat_mask List，包含0代表是否需要光强度，包含1代表是否需要吸收率，包含2代表是否需要反射率
    xseries, yseries = [], []
    index_dict = {}
    idx = 0
    for d in dirs:
        if os.path.isfile(d):
            continue
        files = os.listdir(d)
        fnames = []
        for f in files:
            if f[-6:] == '_r.csv':
                fnames.append(f[:-6])
        tx, ty = [], []
        for fname in fnames:
            refl_name = d + '/' + fname + '_r.csv'
            absb_name = d + '/' + fname + '_a.csv'
            ints_name = d + '/' + fname + '_i.csv'
            xs, rys = _load_csv(refl_name)
            _, ays = _load_csv(absb_name)
            _, iys = _load_csv(ints_name)
            tx.append(xs)
            ty.append([[iy, ay, ry] for ry, ay, iy in zip(rys, ays, iys)])
            index_dict[idx] = d
            idx += 1
        xseries += tx
        yseries += ty
    return np.array(xseries), np.array(yseries)[:, :, feat_mask], index_dict

def _load_np(xarr, yarr, min_wave, max_wave, norm=True):
    series = []
    for tx, ty in zip(xarr, yarr):
        tmp_sery = []
        for xs, ys in zip(tx, ty):
            if xs >= min_wave and xs < max_wave:
                tmp_sery.append(ys)
        if norm:
            tmp_sery = apply_norm(tmp_sery)
        series.append(tmp_sery)
    return np.array(series)

class DataLoader:
    def __init__(self, dir_path, slices=False, norm=True, feat_mask=[1,2]):
        if dir_path[-1] != '/':
            dir_path += '/'
        dirs = [dir_path + lc for lc in os.listdir(dir_path)]
        self.norm = norm
        self.slices = slices
        self.orgx, self.orgy, self.idx_dic = _load_org(dirs, feat_mask=feat_mask)

    def get_patch(self, minx, maxx):
        if not self.slices:
            return _load_np(self.orgx, self.orgy, minx, maxx, norm=self.norm)
        patches = []
        for xs, ys in zip(self.orgx, self.orgy):
            patches.append(_load_np(xs, ys, minx, maxx, norm=self.norm))
        return patches

def load_multivariate(root_dir='D:/0001openCV/01light/nir_pj1/nir_pj1/data/puyan/', minx=890, maxx=1710, mixes=['linen', 'cotton', 'linen_cotton'], device='A', feat_mask=[1,2]):
    train_dirs = root_dir + device + '/'
    test_dirs = root_dir + device + '2/'
    cnts = {}
    for ctrain in os.listdir(train_dirs):
        if ctrain not in cnts:
            cnts[ctrain] = 0
        cnts[ctrain] += len(os.listdir(train_dirs + ctrain))

    for ctest in os.listdir(test_dirs):
        if ctest not in cnts:
            cnts[ctest] = 0
        cnts[ctest] += len(os.listdir(test_dirs + ctest))
    cnames = cnts.keys()
    
    train_x, train_y, test_x, test_y = [], [], [], []
    num_feat = len(feat_mask)
    for cname in cnames:
        if cname not in mixes:
            continue
        train_loader = DataLoader(train_dirs + cname, norm=False, feat_mask=feat_mask)
        cur_trainx = list(train_loader.get_patch(minx, maxx))
        train_loader_norm = DataLoader(train_dirs + cname, feat_mask=feat_mask)
        cur_trainx_norm = list(train_loader_norm.get_patch(minx, maxx))
        for org, norm in zip(cur_trainx, cur_trainx_norm):#org是len=61的数组
            cur_patch = []
            # TODO: 这里加入了三个平滑算法，加上原始数据和规范化数据，一共5个维度
            norg = np.array(org)
            # dsav = list(np.array([signal.savgol_filter(norg[:, i], 5, 3) for i in range(num_feat)]).T)
            dfft = list(np.array([get_fft(norg[:, i]) for i in range(num_feat)]).T)
            # dwie = list(np.array([signal.wiener(norg[:, i]) for i in range(num_feat)]).T)
            # demd = [np.zeros(len(org))]
            # print(dwie)
            demd=[]
            for i in range(num_feat):  # 返回的是imps
                a=norg[:, i]
                imfs=aa(a)
                d = np.sum(imfs, axis=0)  # 返回值是list
                demd.append(d)  # 转置
            demd=list(np.array(demd).T)
            for o, n, f, e in zip(org, norm, dfft, demd):
                cur_patch.append(list(o) + list(n)+ list(f) + list(e))
            # for o, n, s, w in zip(org, norm, dsav, dwie):
            #     cur_patch.append([o, n, s, w])
            train_x.append(cur_patch)

        test_loader = DataLoader(test_dirs + cname, norm=False, feat_mask=feat_mask)
        cur_testx = list(test_loader.get_patch(minx, maxx))
        test_loader_norm = DataLoader(test_dirs + cname, feat_mask=feat_mask)
        cur_testx_norm = list(test_loader_norm.get_patch(minx, maxx))
        for org, norm in zip(cur_testx, cur_testx_norm):
            cur_patch = []
            # TODO: 这里和上面要保持一致
            norg = np.array(org)
            # dsav = list(np.array([signal.savgol_filter(norg[:, i], 5, 3) for i in range(num_feat)]).T)
            dfft = list(np.array([get_fft(norg[:, i]) for i in range(num_feat)]).T)
            # dwie = list(np.array([signal.wiener(norg[:, i]) for i in range(num_feat)]).T)
            demd = []
            for i in range(num_feat):  # 返回的是imps
                a = norg[:, i]
                imfs = aa(a)
                d = np.sum(imfs, axis=0)  # 返回值是list
                demd.append(d)  # 转置
            demd = list(np.array(demd).T)
            for o, n, f, e in zip(org, norm, dfft, demd):
                cur_patch.append(list(o) + list(n) + list(f) + list(e))

            # for o, n, s, w in zip(org, norm, dsav, dwie):
            #     cur_patch.append([o, n, s, w])
            test_x.append(cur_patch)

        if mixes[0] == cname:
            train_y += [0] * len(cur_trainx)
            test_y += [0] * len(cur_testx)
        elif mixes[1] == cname:
            train_y += [1] * len(cur_trainx)
            test_y += [1] * len(cur_testx)
        else:
            train_y += [2] * len(cur_trainx)
            test_y += [2] * len(cur_testx)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

def save_multi_linen_cotton(minx=890, maxx=1710, mixes=['linen', 'cotton', 'linen_cotton'], feat_mask=[1,2]):
    '''
        加载linen cotton及其混合的数据
        TODO：minx, maxx 分别为最小和最大波长，确定输入的波长选择范围，一般为1100-1650以内
                         可以尝试以50为刻度进行调整
    '''
    
    xtrain, ytrain, xtest, ytest = load_multivariate(minx=minx, maxx=maxx, mixes=mixes, device='A', feat_mask=feat_mask)
    print(xtrain.shape, ytrain.shape)
    save_path = 'data/raw/NIR_' + '+'.join(mixes) + '_' + str(minx) + '_' + str(maxx) + '_' + str(feat_mask).replace('[', '').replace(']', '').replace(',', '-').replace(' ', '') + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(save_path + 'X_train.npy', xtrain.transpose(0, 2, 1))
    np.save(save_path + 'y_train.npy', ytrain[:, np.newaxis])
    np.save(save_path + 'X_test.npy', xtest.transpose(0, 2, 1))
    np.save(save_path + 'y_test.npy', ytest[:, np.newaxis])
    print(save_path[9:])

def save_multi_poly_cotton(minx=890, maxx=1710, feat_mask=[1,2]):
    '''
        加载poly cotton及其混合的数据
        TODO：minx, maxx 分别为最小和最大波长，确定输入的波长选择范围，一般为1100-1650以内
                         可以尝试以50为刻度进行调整
    '''
    mixes=['poly', 'cotton', 'poly_cotton']
    xtrain, ytrain, xtest, ytest = load_multivariate(minx=minx, maxx=maxx, mixes=mixes, device='B')
    print(xtrain.shape, ytrain.shape, '///')
    save_path = 'data/raw/NIR_' + '+'.join(mixes) + '_' + str(minx) + '_' + str(maxx) + '_' + str(feat_mask).replace('[', '').replace(']', '').replace(',', '-').replace(' ', '') + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.save(save_path + 'X_train.npy', xtrain.transpose(0, 2, 1))
    np.save(save_path + 'y_train.npy', ytrain[:, np.newaxis])
    np.save(save_path + 'X_test.npy', xtest.transpose(0, 2, 1))
    np.save(save_path + 'y_test.npy', ytest[:, np.newaxis])
    print(save_path[9:])

if __name__ == '__main__':
    # save_multi_linen_cotton(1400, 1600, mixes=['linen', 'cotton'], feat_mask=[1,2])
    save_multi_linen_cotton(1400, 1600, feat_mask=[1,2])
    # save_multi_poly_cotton(1200, 1600, feat_mask=[1,2])