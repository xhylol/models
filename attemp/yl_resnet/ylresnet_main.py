from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

import resnet_model
import resnet_run_loop
from official.utils.flags import core as flags_core
from official.utils.logs import logger

HEIGHT = 270
WIDTH = 480
NUM_CHANNELS = 3
NUM_CLASSES = 103
_NUM_DATA_FILES = 151

NUM_IMAGES = {
    'train': 10000,
    'validation': 10000,
}

DATASET_NAME = 'YL_COVER'

def get_filenames(is_training, data_dir):                                    
  assert tf.io.gfile.exists(data_dir), ('data file does not exist')

  if is_training:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(34, _NUM_DATA_FILES)
    ]
  else:
    return [os.path.join(data_dir, 'data_batch_test.bin')]


def parse_record(raw_record, is_training, dtype):     
  features = tf.compat.v1.parse_single_example(
      raw_record,
      features={
          'img':tf.compat.v1.FixedLenFeature([1],tf.string),
          'cate':tf.compat.v1.FixedLenFeature([1],tf.int64),
          'shape':tf.compat.v1.FixedLenFeature([3],tf.int64)
      })
  cate = tf.reshape(features['cate'],())
  img = tf.io.decode_raw(features['img'], tf.uint8)
  img = tf.reshape(img,features['shape'])
  img = preprocess_image(img, is_training)                     
  img = tf.cast(img, dtype)
  return img, cate

def preprocess_image(image, is_training):
  image = tf.image.resize(image, [HEIGHT, WIDTH])
  if is_training:                                                     
    image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)
  image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])     
  image = tf.image.random_flip_left_right(image)
  image = tf.image.per_image_standardization(image)                     
  return image

def input_fn(is_training,                                               
             data_dir,
             batch_size,
             num_epochs=1,
             dtype=tf.float32,
             datasets_num_private_threads=None,
             parse_record_fn=parse_record,
             input_context=None):

  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.TFRecordDataset(filenames)     # create dataset 

  if input_context:
    tf.compat.v1.logging.info(
        'Sharding the dataset: input_pipeline_id=%d num_input_pipelines=%d' % (
            input_context.input_pipeline_id, input_context.num_input_pipelines))
    dataset = dataset.shard(input_context.num_input_pipelines,
                            input_context.input_pipeline_id)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=NUM_IMAGES['train'],
      parse_record_fn=parse_record_fn,                            
      num_epochs=num_epochs,
      dtype=dtype,
      datasets_num_private_threads=datasets_num_private_threads
  )


def get_synth_input_fn(dtype):
  return resnet_run_loop.get_synth_input_fn(
      HEIGHT, WIDTH, NUM_CHANNELS, NUM_CLASSES, dtype=dtype)

class YLModel(resnet_model.Model):

  def __init__(self, resnet_size, data_format=None, num_classes=NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
  
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(YLModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=None,
        first_pool_stride=None,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )


def _get_block_sizes(resnet_size):
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)

def yl_model_fn(features, labels, mode, params):
  if params['fine_tune']:
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'] * params.get('num_workers', 1),
      batch_denom=256, num_images=NUM_IMAGES['train'],
      boundary_epochs=[30, 60, 80, 90], decay_rates=[1, 0.1, 0.01, 0.001, 1e-4],
      warmup=warmup, base_lr=base_lr)

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=YLModel,
      resnet_size=params['resnet_size'],
      weight_decay=flags.FLAGS.weight_decay,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      fine_tune=params['fine_tune'],
      label_smoothing=flags.FLAGS.label_smoothing
  )    
    
 
def define_yl_flags():
  resnet_run_loop.define_resnet_flags(
                         fp16_implementation=True)
  flags.adopt_module_key_flags(resnet_run_loop)
  flags_core.set_defaults(data_dir='/data/resnet_data/',
                          model_dir='/root/model_dir',
                          train_epochs=90,
                          epochs_between_evals=1,
                          batch_size=128,
                          resnet_size='50',
                          image_bytes_as_serving_input=False,
                          fine_tune=True,
                          pretrained_model_checkpoint_path="/root/tensorflow/resnet_imagenet_v2_20180928",
                          data_format='channels_last',
                          resnet_version='2',
                          export_dir='/root/model_dir/final_export')


def run_yl(flags_obj):
  input_function = (flags_obj.use_synthetic_data and
                    get_synth_input_fn(flags_core.get_tf_dtype(flags_obj)) or
                    input_fn)
  result = resnet_run_loop.resnet_main(
      flags_obj, yl_model_fn, input_function, DATASET_NAME,
      shape=[HEIGHT, WIDTH, NUM_CHANNELS])

  return result


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_yl(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_yl_flags()
  absl_app.run(main)
