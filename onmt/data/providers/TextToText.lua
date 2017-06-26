local Vocabulary = require('onmt.data.Vocabulary')
local TextFileIterator = require('onmt.data.iterators.TextFileIterator')
local TextSplitter = require('onmt.data.transformers.TextSplitter')
local StreamsToIds = require('onmt.data.transformers.StreamsToIds')

local TextToText = torch.class('TextToText')

local options = {
  {
    '-train_src', '',
    [[Path to the training source text data.]],
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
    [[Path to the validation source text data.]],
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
  },
  {
    '-idx_files', false,
    [[If true, source and target files are indexed by an identifier.]]
  }
}

function TextToText.declareOpts(cmd)
  Vocabulary.addOpts(options, 'src', 'source')
  Vocabulary.addOpts(options, 'tgt', 'target')
  cmd:setCmdLineOptions(options)
end

function TextToText:__init(args)
  self.args = args
  self.splitter = TextSplitter.new()
end

function TextToText:getTrainIterators()
  return self:_getIterators(self.args.train_src, self.args.train_tgt)
end
function TextToText:getValidIterators()
  return self:_getIterators(self.args.valid_src, self.args.valid_tgt)
end

function TextToText:_getIterators(src, tgt)
  return {
    TextFileIterator.new(src, self.args.idx_files),
    TextFileIterator.new(tgt, self.args.idx_files)
  }
end

function TextToText:getVocabularies(iterators)
  return {
    Vocabulary.new(self.args, iterators[1], self.splitter, 'src'),
    Vocabulary.new(self.args, iterators[2], self.splitter, 'tgt')
  }
end

function TextToText:getMappers(vocabs)
  local srcTransformer = StreamsToIds.new(vocabs[1].dicts)
  local tgtTransformer = StreamsToIds.new(vocabs[2].dicts)

  return {
    function(item)
      item = self.splitter:transform(item)
      item = srcTransformer:transform(item)
      return item
    end,
    function(item)
      item = self.splitter:transform(item)
      item = tgtTransformer:transform(item)
      return item
    end
  }
end

function TextToText:getParallelValidator()
  return function(entry)
    return not self.args.check_plength or entry[1][1]:size(1) == entry[2][1]:size(1)
  end
end

function TextToText:getValidators()
  return {
    function(item)
      return item[1]:size(1) > 0 and item[1]:size(1) <= self.args.src_seq_length
    end,
    function(item)
      return item[1]:size(1) > 0 and item[1]:size(1) <= self.args.tgt_seq_length
    end
  }
end

function TextToText:sortFunc()
  return function(item)
    return item[1][1]:size(1)
  end
end

return TextToText
