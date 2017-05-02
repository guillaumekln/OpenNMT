local train = {}

train.Trainer = require('onmt.train.Trainer')
train.ParallelTrainer = require('onmt.train.ParallelTrainer')
train.SynchronousParallelTrainer = require('onmt.train.SynchronousParallelTrainer')
train.AsynchronousParallelTrainer = require('onmt.train.AsynchronousParallelTrainer')
train.Saver = require('onmt.train.Saver')
train.EpochState = require('onmt.train.EpochState')
train.Optim = require('onmt.train.Optim')

return train
