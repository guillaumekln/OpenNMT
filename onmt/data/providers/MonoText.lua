local Vocabulary = require('onmt.data.Vocabulary')
local TextFileIterator = require('onmt.data.iterators.TextFileIterator')
local TextSplitter = require('onmt.data.transformers.TextSplitter')
local StreamsToIds = require('onmt.data.transformers.StreamsToIds')

local MonoText = torch.class('MonoText')

local options = {
  {
    '-train', '',
    [[Path to the training text data.]],
    {
      valid = ExtendedCmdLine.fileExists
    }
  },
  {
    '-valid', '',
    [[Path to the validation text data.]],
    {
      valid = ExtendedCmdLine.fileExists
    }
  },
  {
    '-seq_length', 50,
    [[Maximum sequence length.]],
    {
      valid = ExtendedCmdLine.isInt(1)
    }
  }
}

function MonoText.declareOpts(cmd)
  Vocabulary.addOpts(options)
  cmd:setCmdLineOptions(options)
end

function MonoText:__init(args)
  self.args = args
  self.splitter = TextSplitter.new()
end

function MonoText:getTrainIterators()
  return {
    TextFileIterator.new(self.args.train)
  }
end

function MonoText:getValidIterators()
  return {
    TextFileIterator.new(self.args.valid)
  }
end

function MonoText:getVocabularies(iterators)
  return {
    Vocabulary.new(self.args, iterators[1], self.splitter),
  }
end

function MonoText:getMappers(vocabs)
  local transformer = StreamsToIds.new(vocabs[1].dicts)

  return {
    function(item)
      item = self.splitter:transform(item)
      item = transformer:transform(item)
      return item
    end
  }
end

function MonoText:getValidators()
  return {
    function(item)
      return item[1]:size(1) > 0 and item[1]:size(1) <= self.args.seq_length
    end
  }
end

function MonoText:sortFunc()
  return function(item)
    return item[1][1]:size(1)
  end
end

return MonoText
