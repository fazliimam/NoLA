import unittest
import torch
import numpy as np

def select_top_k_similarity_per_class(outputs, img_paths, K=1, image_features=None, is_softmax=True):
    # print(outputs.shape)
    if is_softmax:
        outputs = torch.nn.Softmax(dim=1)(outputs)
    output_m = outputs.cpu().detach().numpy()
    output_ori = outputs.cpu().detach()
    output_m_max = output_m.max(axis=1)
    output_m_max_id = np.argsort(-output_m_max)
    output_m = output_m[output_m_max_id]
    img_paths = img_paths[output_m_max_id]
    output_m_max = output_m_max[output_m_max_id]
    output_ori = output_ori[output_m_max_id]
    ids = (-output_m).argsort()[:, 0] # 获得每行的类别标签
 
    if image_features is not None:
        image_features = image_features.cpu().detach()
        image_features = image_features[output_m_max_id]
 
    predict_label_dict = {}
    predict_conf_dict = {}
    from tqdm import tqdm
    for id in tqdm(list(set(ids.tolist()))): # 标签去重
        index = np.where(ids==id)
        conf_class = output_m_max[index] # 置信度
        output_class = output_ori[index]
        img_paths_class = img_paths[index] # 每个类别的路径
 
        if image_features is not None:
            img_features = image_features[index]
            if K >= 0:
                for img_path, img_feature, conf, logit in zip(img_paths_class[:K], img_features[:K], conf_class[:K], output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
            else:
                for img_path, img_feature, conf, logit in zip(img_paths_class, img_features, conf_class, output_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = [id, img_feature, conf, logit]
        else:
            if K >= 0:
                for img_path, conf in zip(img_paths_class[:K], conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
            else:
                for img_path, conf in zip(img_paths_class, conf_class):
                    if '/data/' in img_path:
                        img_path = './data/' + img_path.split('/data/')[1]
                    predict_label_dict[img_path] = id
                    predict_conf_dict[img_path] = conf
    return predict_label_dict, predict_conf_dict

class TestLaFTer(unittest.TestCase):
    def setUp(self):
        self.outputs = torch.randn(100, 20)  # 100 samples, 20 classes
        self.img_paths = np.array(['path' + str(i) for i in range(100)])
        self.K = 16
        self.image_features = torch.randn(100, 20)
        self.is_softmax = False

    def test_select_top_k_similarity_per_class(self):
        predict_label_dict, predict_conf_dict = select_top_k_similarity_per_class(self.outputs, self.img_paths, self.K, self.image_features, self.is_softmax)

        # Check if each sample can be mapped to one class
        for img_path, value in predict_label_dict.items():
            self.assertIsInstance(value, list)
            self.assertEqual(len(value), 4)  # id, img_feature, conf, logit
            self.assertIsInstance(value[0], int)  # id
            self.assertIsInstance(value[1], torch.Tensor)  # img_feature
            self.assertIsInstance(value[2], np.float32)  # conf
            self.assertIsInstance(value[3], torch.Tensor)  # logit

        # Check if each sample has a confidence score
        for img_path, conf in predict_conf_dict.items():
            self.assertIsInstance(conf, np.float32)

if __name__ == '__main__':
    unittest.main()