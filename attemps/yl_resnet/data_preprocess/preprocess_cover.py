import logging
import random 
import io
import os
import requests
from impala.dbapi import connect
from PIL import Image
import tensorflow as tf
import numpy as np
import multiprocessing as mp
import math
import thread

logger = logging.getLogger("tensorflow")
logger.setLevel(logging.INFO)
logger.propagate = False
fh = logging.FileHandler('/home/rec/xhy/nas/resnet_data/process.log','w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

Image.MAX_IMAGE_PIXELS = 1000000000

def serialize_example_pyfunction(cate, img, shape):
    feature = {
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        'cate': tf.train.Feature(int64_list=tf.train.Int64List(value=[cate])),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

class HiveClient:
     def __init__(self, db_host, port, authMechanism, user, password, database):
         self.conn = connect(host=db_host,
                             port=port,
                             auth_mechanism=authMechanism,
                             user=user,
                             password=password,
                             database=database)
 
     def query(self, sql):
         with self.conn.cursor() as cursor:
             cursor.execute(sql)
             return cursor.fetchall()
 
     def close(self):
         self.conn.close()

def get_data_from_hive():
    hive_client = HiveClient(db_host='localhost',
                             port=10000,
                             authMechanism='PLAIN',
                             user='niehuanghui',
                             password='nhg!yl888',
                             database='rec_db')

    sql = "select img,category_id from basis_rdb.basis_fe_video where img is not null and img is not null and (audit_status=1 OR choice>4) and disable = 0"
    datas = hive_client.query(sql)
    f = open('./img_data.csv','wr')
    for data in datas:
        f.write(','.join([str(d) for d in data]) + '\n')
    f.close()
    return datas

def get_data_from_file():
    f = open('./img_data.csv')
    datas = []
    for eachline in f:
        line = eachline.replace('\n','').split(',')
        datas.append([line[0],int(line[1])])
    return datas

def get_data():
    if os.path.exists('./img_data.csv'):
        return get_data_from_file()
    return get_data_from_hive()


def create_test_tf_record(queue):
    writer= tf.io.TFRecordWriter("data_batch_test.bin")
    while 1:
        serialize_example = queue.get()
        if serialize_example == 'kill':
            break
        writer.write(serialize_example)
    writer.close()

def create_train_tf_record(datas,cate_dict,index,queue):
    writer = tf.io.TFRecordWriter("data_batch_{}.bin".format(index))
    logger.info('{} part start.'.format(index))
    count = 0
    for data in datas:
        if data[0] == '':
            continue
        img_url = "https://img.yilan.tv/" + data[0]
        cate = data[1]
        r = requests.get(img_url)
        try:
            img = Image.open(io.BytesIO(r.content)).convert('RGB')
        except Exception as e:
            logger.info("Error {}, invalid pic, url: {}".format(e,img_url))
            continue
        try:
            shape = list(np.array(img).shape)
            img = img.tobytes()
            if cate not in cate_dict:
                cate_dict[cate] = len(cate_dict)
            if random.uniform(0,10) < 0.07:
                queue.put(serialize_example_pyfunction(cate_dict[cate],img,shape))
            else:
                count += 1
                writer.write(serialize_example_pyfunction(cate_dict[cate],img,shape))
        except Exception as e:
            logger.info(e)
            continue
    logger.info('{} part finish, total count {}.'.format(index,count))
    writer.close()

def main():
    manager = mp.Manager()
    queue = manager.Queue()

    cate_dict = manager.dict() 
    pool = mp.Pool(15)

    #pool.apply_async(create_test_tf_record, (queue,))
    thread.start_new_thread(create_test_tf_record,(queue,))

    datas = get_data()
    random.shuffle(datas)
    start = 0
    end = 0
    step = 10000
    while 1:
        start = end
        end = min(start + step,len(datas))
        pool.apply_async(create_train_tf_record, (datas[start:end],cate_dict,start/step,queue,))
        if end == len(datas):
            break
    pool.close()
    pool.join() 
    queue.put('kill')
    logger.info(cate_dict)

if __name__ == "__main__":
    main()
