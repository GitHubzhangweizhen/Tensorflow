import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path,uid_lookup_path)
    
    def load(self,label_lookup_path,uid_lookup_path):
        #加载分类字符串n********对应分类名的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        for line in proto_as_ascii_lines:
            #去掉换行符
            line = line.strip('\n')
            #按照‘\t’分割
            parsed_items = line.split('\t')
            #获取分类编号
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[1]
            #保存编号n*******与分类名称的对应关系
            uid_to_human[uid] = human_string
        
        #加载分类字符串n*******对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类编号1-1000
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                #获取编号字符串n**********
                target_class_string = line.split(': ')[1]
                #保存分类编号1-1000与编号字符串n********的映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]
        
        #建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key,val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            #建立分类编号1-1000到分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name
    #传入分类编号1-1000的返回分类名称
    def id_to_string(self,node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]
        
#创建一个图来存放google训练好的模型
with tf.gfile.GFile('inception_model/classify_image_graph_def.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')
    
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #遍历目录
    for root,dirs,files in os.walk('images/'):
        for file in files:
            #载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root,file),'rb').read()
            predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})  #图片格式是jpg
            predictions = np.squeeze(predictions) #把结果转化为1维数据
            
            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
            #排序
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:
                #获取分类名称
                human_string =node_lookup.id_to_string(node_id)
                #获取该分类的置信度
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string,score))
            print()
    
    
images/car.jpg

minivan (score = 0.84354)
beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon (score = 0.02371)
cab, hack, taxi, taxicab (score = 0.01454)
car wheel (score = 0.00736)
pickup, pickup truck (score = 0.00494)

images/cat.jpeg

tabby, tabby cat (score = 0.48859)
tiger cat (score = 0.44788)
Egyptian cat (score = 0.01485)
bath towel (score = 0.00205)
Persian cat (score = 0.00134)

images/dog.jpg

cocker spaniel, English cocker spaniel, cocker (score = 0.44338)
Blenheim spaniel (score = 0.25386)
Welsh springer spaniel (score = 0.05874)
Sussex spaniel (score = 0.02439)
clumber, clumber spaniel (score = 0.00785)

images/eagle.jpg

bald eagle, American eagle, Haliaeetus leucocephalus (score = 0.92000)
kite (score = 0.00593)
vulture (score = 0.00099)
magpie (score = 0.00070)
hen (score = 0.00064)

images/seaside.jpeg

seashore, coast, seacoast, sea-coast (score = 0.66978)
sandbar, sand bar (score = 0.14855)
lakeside, lakeshore (score = 0.01674)
promontory, headland, head, foreland (score = 0.01443)
coral reef (score = 0.01215)

 