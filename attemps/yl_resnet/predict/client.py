from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import grpc
from impala.dbapi import connect
import os
import requests
from PIL import Image
import numpy as np
import logging
import io
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.python.framework import tensor_util
from threading import Thread
import multiprocessing as mp


logger = logging.getLogger("tensorflow")
logger.setLevel(logging.INFO)
logger.propagate = False
fh = logging.FileHandler('./process.log','w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

Image.MAX_IMAGE_PIXELS = 1000000000
HEIGHT = 270
WIDTH = 480
NUM_CHANNELS = 3
VEC_SIZE = 128

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

    sql = "select video_id, img from basis_rdb.basis_fe_video where img is not null and img <> '' and (audit_status=1 OR choice>4) and disable = 0"
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
        datas.append(line)
    return datas

def get_data():
    if os.path.exists('./img_data.csv'):
        return get_data_from_file()
    return get_data_from_hive()

def do_inference(hostport,data_queue,write_queue):
    logger.info("start the inference process.")
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'

    def write_result(videoids,inputs):
        logger.info("inputs shape is {}.".format(inputs.shape))
        request.inputs['input'].CopyFrom(tensor_util.make_tensor_proto(inputs, shape=inputs.shape, dtype=tf.float32))
        result = stub.Predict(request, 100)
        res = result.outputs['embedding'].float_val
        if len(res) != len(videoids) * 128:
            logger.info("res length is {}, but videoids length is {}.".format(len(res), len(videoids)))
            return
        else:
            for i in range(len(videoids)):
                write_queue.put('{}:{}\n'.format(videoids[i],','.join([str(x) for x in res[i*128:(i+1)*128]])))
            logger.info("put {} elements to the shared queue.".format(len(videoids)))

    while True:
        videoids = []
        inputs = []
        for i in range(100):
            try:
                data = data_queue.get(block=True, timeout=2)
            except:
                continue
            if data == 'kill':
                break
            videoids.append(data[0])
            inputs.append(data[1])
        try:
            inputs = np.array(inputs)
            write_result(videoids,inputs)
        except Exception as e:
             logger.info(e)
             continue


def download_cover(datas,data_queue):
    logger.info("start download process.")
    for i in range(len(datas)):
        data = datas[i]
        videoid = data[0]
        img_url = "https://img.yilan.tv/" + data[1]
        try:
            r = requests.get(img_url,timeout=(5, 10))
            img = np.array(Image.open(io.BytesIO(r.content)).resize((WIDTH,HEIGHT),Image.BILINEAR).convert('RGB'))
            data_queue.put([videoid,img])
        except Exception as e:
            logger.info("Error {}, invalid pic, url: {}.".format(e,img_url))

def write_embedding(write_queue):
    with open('./video_embedding.csv','w') as f:
        while 1:
            id_embedding = write_queue.get()
            if id_embedding == 'kill':
                break
            f.write(id_embedding)

def main():
    grpc_url = '172.17.105.157:31873'
    manager = mp.Manager()

    write_queue = manager.Queue()
    data_queue = manager.Queue()

    inference_pool = mp.Pool(1)
    download_pool = mp.Pool(6)

    datas = get_data()
    logger.info("read data done.")

    thread = Thread(target=write_embedding, args=(write_queue,))
    thread.start()

    step_size = int(len(datas)/6)
    for i in range(0, len(datas), step_size):
        download_pool.apply_async(download_cover, (datas[i:i+step_size],data_queue,))
    download_pool.close()

    inference_pool.apply_async(do_inference,(grpc_url,data_queue,write_queue,))
    inference_pool.close()

    logger.info("start to wait for the download process to complete.")
    download_pool.join()
    data_queue.put('kill')
    logger.info("start to wait for the inference process to complete.")
    inference_pool.join()
    write_queue.put('kill')

    thread.join()

if __name__ == "__main__":
    main()
