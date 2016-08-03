# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to train Inception using multiple replicas with synchronous update.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline


from inception import image_processing, compute_group_optimizer
from inception import inception_model as inception
from inception.slim import slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/lfs/local/0/daniter/optimizer',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10*60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 30000,
                            'Save summaries interval seconds.')
tf.app.flags.DEFINE_integer('compute_groups', 1,
                            'Number of compute groups. Worers are divided equally.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,#0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10000.0,#2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 1.0,#0.94,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_float('momentum', 0.9,'Momentum term')
tf.app.flags.DEFINE_boolean('sync', True, "Async Mode")


# ******************************* OPTIMIZER PARAMETERS ***********************

tf.app.flags.DEFINE_integer('duration', 1*60, "How long should the run last in seconds")
tf.app.flags.DEFINE_string('checkpoint_prefix', "model.ckpt", "Name to use for the checkpoint files")



# ******************************* END OPTIMIZER PARAMETERS ***********************

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


def train(target, dataset, cluster_spec):
  """Train Inception on a dataset for a number of steps."""
  # Number of workers and parameter servers are infered from the workers and ps
  # hosts string.

  tf.logging.info("Starting: %s" % datetime.now() )
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  # If no value is given, num_replicas_to_aggregate defaults to be the number of
  # workers.
  if FLAGS.num_replicas_to_aggregate == -1:
    num_replicas_to_aggregate = num_workers
  else:
    num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

  # Both should be greater than 0 in a distributed training.
  assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                         'num_parameter_servers'
                                                         ' must be > 0.')

  # Choose worker 0 as the chief. Note that any worker could be the chief
  # but there should be only one chief.
  is_chief = (FLAGS.task_id == 0)

  if num_workers == FLAGS.compute_groups:
    FLAGS.sync = False

  # daniter - compute groups:
  assert num_workers % FLAGS.compute_groups == 0, ("Number of workers msut be divisible by compute groups")
  is_cg_primary = (FLAGS.task_id < FLAGS.compute_groups)


  # Ops are assigned to worker by default.
  with tf.device('/job:worker/task:%d' % FLAGS.task_id):
    tf.set_random_seed(FLAGS.DANITER_SEED)
    # Variables and its related init/assign ops are assigned to ps.
    with slim.scopes.arg_scope(
        [slim.variables.variable, slim.variables.global_step],
        device=slim.variables.VariableDeviceChooser(num_parameter_servers)):
      # Create a variable to count the number of train() calls. This equals the
      # number of updates applied to the variables.
      global_step = slim.variables.global_step()

      # Calculate the learning rate schedule.
      num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                               FLAGS.batch_size)
      # Decay steps need to be divided by the number of replicas to aggregate.
      decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay /
                        num_replicas_to_aggregate)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True)
      # Add a summary to track the learning rate.
      tf.scalar_summary('learning_rate', lr)

      # Create an optimizer that performs gradient descent.
      #opt = tf.train.RMSPropOptimizer(lr,
    #                                  RMSPROP_DECAY,
#                                      momentum=RMSPROP_MOMENTUM,
    #                                  epsilon=RMSPROP_EPSILON)

      opt = tf.train.MomentumOptimizer(lr,FLAGS.momentum,use_locking=True) # Tuning done for these!
      tf.logging.info("Learning rate: %f, momentum: %f" % (FLAGS.initial_learning_rate, FLAGS.momentum))

      images, labels = image_processing.distorted_inputs(
          dataset,
          batch_size=FLAGS.batch_size,
          num_preprocess_threads=FLAGS.num_preprocess_threads)

      # Number of classes in the Dataset label set plus 1.
      # Label 0 is reserved for an (unused) background class.
      num_classes = dataset.num_classes() + 1

      #with tf.control_dependencies([tf.Print(images, tf.split(0, 16, images), "images:", summarize=2)]):
      logits = inception.inference(images, num_classes, for_training=True)

      #with tf.control_dependencies([tf.Print(logits[0], [logits[0]], "logits", summarize=10)]):
        # Add classification loss.
      inception.loss(logits, labels)

      #Accuracy
      #correct_prediction = tf.nn.in_top_k(logits[0], labels, 1)
      #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

      # Gather all of the losses including regularization losses.
      losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
      losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

      #with tf.control_dependencies([tf.Print(losses, [losses], "pre-addn losses", summarize=64)]):
      total_loss = tf.add_n(losses, name='total_loss')

      if is_chief:
        # Compute the moving average of all individual losses and the
        # total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summmary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
          loss_name = l.op.name
          # Name each loss as '(raw)' and name the moving average version of the
          # loss as the original loss name.
          tf.scalar_summary(loss_name + ' (raw)', l)
          tf.scalar_summary(loss_name, loss_averages.average(l))

        # Add dependency to compute loss_averages.
        with tf.control_dependencies([loss_averages_op]):
          total_loss = tf.identity(total_loss)
          #total_loss = tf.Print(total_loss, [total_loss], "first total loss:")


      # Track the moving averages of all trainable variables.
      # Note that we maintain a 'double-average' of the BatchNormalization
      # global statistics.
      # This is not needed when the number of replicas are small but important
      # for synchronous distributed training with tens of workers/replicas.
      exp_moving_averager = tf.train.ExponentialMovingAverage(
          inception.MOVING_AVERAGE_DECAY, global_step)

      variables_to_average = (
          tf.trainable_variables() + tf.moving_average_variables())

      # Add histograms for model variables.
      for var in variables_to_average:
        tf.histogram_summary(var.op.name, var)

      # Create synchronous replica optimizer.
      if FLAGS.sync:
          tf.logging.info("Sync mode!!!!!!")
          tf.logging.info("Compute groups : %d" % FLAGS.compute_groups)
          opt = compute_group_optimizer.ComputeGroupOptimizer(
              opt,
              replicas_to_aggregate=num_replicas_to_aggregate,
              replica_id=FLAGS.task_id,
              total_num_replicas=num_workers,
              variable_averages=exp_moving_averager,
              variables_to_average=variables_to_average)

      batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
      assert batchnorm_updates, 'Batchnorm updates are missing'
      batchnorm_updates_op = tf.group(*batchnorm_updates)
      # Add dependency to compute batchnorm_updates.
      with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(total_loss)

      #total_loss = tf.Print(total_loss, [total_loss], "total loss:")

      # Compute gradients with respect to the loss.
      grads = opt.compute_gradients(total_loss, gate_gradients=2)

      #grads = [(tf.clip_by_value(grad, -1, 1),v) for grad,v in grads]


      # Add histograms for gradients.
      for grad, var in grads:
        if grad is not None:
          tf.histogram_summary(var.op.name + '/gradients', grad)

      apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)


      with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.identity(total_loss, name='train_op')

      # Get chief queue_runners, init_tokens and clean_up_op, which is used to
      # synchronize replicas.
      # More details can be found in sync_replicas_optimizer.
      if FLAGS.sync:
          chief_queue_runners = [opt.get_chief_queue_runner()]
          init_tokens_op = opt.get_init_tokens_op()
          clean_up_op = opt.get_clean_up_op()
          start_op = opt.start()
          cheif_start_op = opt.cheif_starter()

      # Create a saver.
      saver = tf.train.Saver(max_to_keep=100)

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.merge_all_summaries()

      # Build an initialization operation to run below.
      init_op = tf.initialize_all_variables()

      # We run the summaries in the same thread as the training operations by
      # passing in None for summary_op to avoid a summary_thread being started.
      # Running summaries and training operations in parallel could run out of
      # GPU memory.
      sv = tf.train.Supervisor(is_chief=is_chief, # TODO : should this be per cheif?
                               logdir=FLAGS.train_dir,
                               init_op=init_op,
                               summary_op=None,
                               global_step=global_step,
                               saver=saver,
                               save_model_secs=0)#FLAGS.save_interval_secs)

      tf.logging.info('%s Supervisor' % datetime.now())

      sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=FLAGS.log_device_placement)

      # Get a session.
      sess = sv.prepare_or_wait_for_session(target, config=sess_config)

      tf.logging.info("got sessions! %s " % datetime.now())

      # Start the queue runners.
      queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
      sv.start_queue_runners(sess, queue_runners)
      tf.logging.info('Started %d queues for processing input data.',
                      len(queue_runners))

      if is_chief and FLAGS.sync:
        sess.run(cheif_start_op)

      if FLAGS.sync:
        sess.run(start_op)

      if is_cg_primary and FLAGS.sync:
        sv.start_queue_runners(sess, chief_queue_runners)
        sess.run(init_tokens_op)

      # Train, checking for Nans. Concurrently run the summary operation at a
      # specified interval. Note that the summary_op and train_op never run
      # simultaneously in order to prevent running out of GPU memory.
      tf.set_random_seed(FLAGS.DANITER_SEED)
      next_summary_time = time.time() + FLAGS.save_summaries_secs

      begin_run = time.time()


      while not sv.should_stop() and (time.time()-begin_run < FLAGS.duration):

        try:
          tf.logging.info("Actual first run: %s" % datetime.now())
          start_time = time.time()
          loss_value, step = sess.run([train_op, global_step])
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
          if step > FLAGS.max_steps:
            break
          duration = time.time() - start_time

          #Print Accuracy
          #tf.logging.info("Step: %d, Accuracy: %f, Loss: %f, time: %.3f sec/batch" %(step, sess.run(accuracy), loss_value, duration))
          tf.logging.info("Step: %d, Loss: %f, time: %.3f sec/batch" %(step, loss_value, duration))

          # if False:
          #   examples_per_sec = FLAGS.batch_size / float(duration)
          #   format_str = ('Worker %d: %s: step %d, loss = %.2f'
          #                 '(%.1f examples/sec; %.3f  sec/batch)')
          #   tf.logging.info(format_str %
          #                   (FLAGS.task_id, datetime.now(), step, loss_value,
          #                    examples_per_sec, duration))

          # Determine if the summary_op should be run on the chief worker.
          #if is_chief and next_summary_time < time.time():
          #  tf.logging.info('Running Summary operation on the chief.')
          #  summary_str = sess.run(summary_op)
          #  sv.summary_computed(sess, summary_str)
          #  tf.logging.info('Finished running Summary operation.')

            # Determine the next time for running the summary.
           # next_summary_time += FLAGS.save_summaries_secs
        except:
          if is_chief:
            tf.logging.info('About to execute sync_clean_up_op!')
            sess.run(clean_up_op)
          raise

      # Stop the supervisor.  This also waits for service threads to finish.
      try:
        sv.stop()
      except:
        pass

      # Save after the training ends.
      if is_chief:
        saver.save(sess,
                  os.path.join(FLAGS.train_dir, FLAGS.checkpoint_prefix),
                  global_step=global_step)
        tf.logging.info("finished writing checkpoint")
