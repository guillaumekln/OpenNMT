local Vocabulary = require('onmt.data.Vocabulary')
local TextFileIterator = require('onmt.data.iterators.TextFileIterator')
local TextSplitter = require('onmt.data.transformers.TextSplitter')

local ExtendedCmdLine = require('onmt.utils.ExtendedCmdLine')
local Logger = require('onmt.utils.Logger')


local options = {
  {
    '-data', '',
    [[Data file.]],
    {
      valid = ExtendedCmdLine.fileExists
    }
  },
  {
    '-save_vocab', '',
    [[Vocabulary dictionary prefix.]],
    {
      valid = ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-idx_files', false,
    [[If set, each line of the data file starts with a first field which is the index of the sentence.]]
  }
}

Vocabulary.addOpts(options)

local cmd = ExtendedCmdLine.new('build_vocab.lua')
cmd:setCmdLineOptions(options, 'Vocabulary')
Logger.declareOpts(cmd)

local opt = cmd:parse(arg)

_G.logger = Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

local iterator = TextFileIterator.new(opt.data, opt.idx_files)
local splitter = TextSplitter.new()
local vocab = Vocabulary.new(opt, iterator, splitter)

vocab:save(opt.save_vocab)

_G.logger:shutDown()
