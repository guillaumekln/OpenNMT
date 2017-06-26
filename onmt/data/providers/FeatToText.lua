local Vocabulary = require('onmt.data.Vocabulary')
local TextFileIterator = require('onmt.data.iterators.TextFileIterator')
local KaldiFileIterator = require('onmt.data.iterators.KaldiFileIterator')
local TextSplitter = require('onmt.data.transformers.TextSplitter')
local StreamsToIds = require('onmt.data.transformers.StreamsToIds')

local FeatToText = torch.class('FeatToText')

local options = {
  {
    '-train_src', '',
    [[Path to the training source data in the Kaldi format.]],
    {
      valid = ExtendedCmdLine.fileExists
    }
  },
  {
    '-train_tgt', '',
    [[Path to the training target text data.]],
    {
      valid = ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid_src', '',
    [[Path to the validation source data in the Kaldi format.]],
    {
      valid = ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid_tgt', '',
    [[Path to the validation target text data.]],
    {
      valid = ExtendedCmdLine.fileExists
    }
  },
  {
    '-src_seq_length', 50,
    [[Maximum source sequence length.]],
    {
      valid = ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-tgt_seq_length', 50,
    [[Maximum target sequence length.]],
    {
      valid = ExtendedCmdLine.isInt(1)
    }
  },
  {
    '-check_plength', false,
    [[Check that source and target have same length (e.g. for sequence tagging).]]
  }
}

function FeatToText.declareOpts(cmd)
  Vocabulary.addOpts(options, 'tgt', 'target')
  cmd:setCmdLineOptions(options)
end

function FeatToText:__init(args)
  self.args = args
  self.splitter = TextSplitter.new()
end

function FeatToText:getTrainIterators()
  return self:_getIterators(self.args.train_src, self.args.train_tgt)
end
function FeatToText:getValidIterators()
  return self:_getIterators(self.args.valid_src, self.args.valid_tgt)
end

function FeatToText:_getIterators(src, tgt)
  return {
    KaldiFileIterator.new(src),
    TextFileIterator.new(tgt, true)
  }
end

function FeatToText:getVocabularies(iterators)
  return {
    Vocabulary.new(self.args, iterators[2], self.splitter, 'tgt')
  }
end

function FeatToText:getMappers(vocabs)
  local tgtTransformer = StreamsToIds.new(vocabs[1].dicts)

  return {
    function(item)
      return item
    end,
    function(item)
      item = self.splitter:transform(item)
      item = tgtTransformer:transform(item)
      return item
    end
  }
end

function FeatToText:getParallelValidator()
  return function(entry)
    return not self.args.check_plength or entry[1]:size(1) == entry[2][1]:size(1)
  end
end

function FeatToText:getValidators()
  return {
    function(item)
      return item:size(1) > 0 and item:size(1) <= self.args.src_seq_length
    end,
    function(item)
      return item[1]:size(1) > 0 and item[1]:size(1) <= self.args.tgt_seq_length
    end
  }
end

function FeatToText:sortFunc()
  return function(item)
    return item[1]:size(1)
  end
end

return FeatToText
