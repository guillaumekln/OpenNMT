require('onmt.init')

local Vocabulary = require('onmt.data.Vocabulary')
local KaldiFileIterator = require('onmt.data.KaldiFileIterator')
local TextFileIterator = require('onmt.data.TextFileIterator')
local TextSplitter = require('onmt.data.TextSplitter')
local StreamsToIds = require('onmt.data.StreamsToIds')
local InMemoryDataset = require('onmt.data.InMemoryDataset')

local Table = require('onmt.utils.Table')
local Logger = require('onmt.utils.Logger')
local ExtendedCmdLine = require('onmt.utils.ExtendedCmdLine')

-- Options declaration.
local options = {
  {
    '-data_type', 'bitext',
    [[Type of data to preprocess. Use 'monotext' for monolingual data.
      This option impacts all options choices.]],
    {
      enum = {'bitext', 'monotext', 'feattext'}
    }
  },
  {
    '-save_data', '',
    [[Output file for the prepared data.]],
    {
      valid = ExtendedCmdLine.nonEmpty
    }
  },
  {
    '-sort', true,
    [[If set, sort the sequences by size to build batches without source padding.]]
  },
  {
    '-shuffle', true,
    [[If set, shuffle the data (prior sorting).]]
  },
  {
    '-idx_files', false,
    [[If set, source and target files are 'key value' with key match between source and target.]]
  },
  {
    '-check_plength', false,
    [[Check source and target have same length (for seq tagging).]]
  },
  {
    '-report_every', 100000,
    [[Report status every this many sentences.]],
    {
      valid = ExtendedCmdLine.isInt(1)
    }
  }
}

local otherOptions = {
  {
    '-seed', 3425,
    [[Random seed.]],
    {
      valid = ExtendedCmdLine.isUInt()
    }
  }
}

--[[ Adds prefixed options. ]]
local function addOpts(options, prefix, name)
  prefix = prefix and '_' .. prefix or ''
  name = name or ''

  Table.append(
    options,
    {
      {
        '-train' .. prefix, '',
        'Path to the training ' .. name.. ' data.',
        {
          valid = ExtendedCmdLine.fileExists
        }
      },
      {
        '-valid' .. prefix, '',
        'Path to the validation ' .. name.. ' data.',
        {
          valid = ExtendedCmdLine.fileExists
        }
      },
      {
        '-' .. prefix ..'seq_length', 50,
        'Maximum ' .. name .. ' sequence length.',
        {
          valid = ExtendedCmdLine.isInt(1)
        }
      }
    }
  )

  return options
end


local cmd = ExtendedCmdLine.new('preprocess.lua')

-- Get the data type to determine other options.
local dataType = cmd.getArgument(arg, '-data_type') or 'bitext'

if dataType == 'bitext' or dataType == 'feattext' then
  if dataType == 'bitext' then
    Vocabulary.addOpts(options, 'src', 'source')
  end

  Vocabulary.addOpts(options, 'tgt', 'target')
  addOpts(options, 'src', 'source')
  addOpts(options, 'tgt', 'target')
elseif dataType == 'monotext' then
  Vocabulary.addOpts(options)
  addOpts(options)
end

cmd:setCmdLineOptions(options, 'Preprocess')
cmd:setCmdLineOptions(otherOptions, 'Other')
Logger.declareOpts(cmd)

-- local function isValid(seq, maxSeqLength)
--   if torch.isTensor(seq) then
--     return seq:size(1) > 0 and seq:size(1) <= maxSeqLength
--   end
--   return #seq > 0 and #seq <= maxSeqLength
-- end

-- local function parallelCheck(idx, _, _, tokens)
--   local length1 = (type(tokens[1])=='table' and #tokens[1]) or (tokens[1]:dim()==0 and 0) or tokens[1]:size(1)
--   local length2 = (type(tokens[2])=='table' and #tokens[2]) or (tokens[2]:dim()==0 and 0) or tokens[2]:size(1)
--   if length1~=length2 then
--     _G.logger:warning('SENT %s: source/target not aligned (%d/%d)', tostring(idx), length1, length2)
--     return false
--   end
--   return true
-- end

local function main()
  local opt = cmd:parse(arg)

  torch.manualSeed(opt.seed)

  _G.logger = Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local fileIterators = {}
  local vocabs = {}
  local mappers = {}
  local validators = {}

  local splitter = TextSplitter.new()

  if dataType == 'bitext' then
    fileIterators[1] = TextFileIterator.new(opt.train_src, opt.idx_files)
    fileIterators[2] = TextFileIterator.new(opt.train_tgt, opt.idx_files)

    vocabs[1] = Vocabulary.new(opt, fileIterators[1], splitter, 'src')
    vocabs[2] = Vocabulary.new(opt, fileIterators[2], splitter, 'tgt')

    local srcTransformer = StreamsToIds.new(vocabs[1].dicts)
    local tgtTransformer = StreamsToIds.new(vocabs[2].dicts)

    mappers[1] = function (item)
      item = splitter:transform(item)
      item = srcTransformer:transform(item)
      return item
    end
    mappers[2] = function (item)
      item = splitter:transform(item)
      item = tgtTransformer:transform(item)
      return item
    end
  elseif dataType == 'feattext' then
    fileIterators[1] = KaldiFileIterator.new(opt.train_src)
    fileIterators[2] = TextFileIterator.new(opt.train_tgt, true)

    vocabs[1] = Vocabulary.new(opt, fileIterators[2], splitter, 'tgt')

    local tgtTransformer = StreamsToIds.new(vocabs[2].dicts)

    mappers[1] = function (item)
      return item
    end
    mappers[2] = function (item)
      item = splitter:transform(item)
      item = tgtTransformer:transform(item)
      return item
    end
  elseif dataType == 'monotext' then
    fileIterators[1] = TextFileIterator.new(opt.train)

    vocabs[1] = Vocabulary.new(opt, fileIterators[1], splitter)

    local transformer = StreamsToIds.new(vocabs[1].dicts)

    mappers[1] = function (item)
      item = splitter:transform(item)
      item = transformer:transform(item)
      return item
    end
  end

  for i = 1, #fileIterators do
    fileIterators[i]:reset()
  end

  local dataset = InMemoryDataset.new()

  dataset = dataset:map(mappers)
  dataset = dataset:fromFiles(fileIterators)

  if opt.shuffle then
    dataset = dataset:shuffle()
  end

  if opt.sort then
    dataset = dataset:sort(function (item)
      return item[1][1]:size(1)
    end)
  end

  -- local Vocabulary = onmt.data.Vocabulary
  -- local Preprocessor = onmt.data.Preprocessor.new(opt, dataType)

  -- local data = { dataType=dataType }

  -- -- keep processing options in the structure for further traceability
  -- data.opt = opt

  -- data.dicts = {}

  -- _G.logger:info('Preparing vocabulary...')
  -- if dataType ~= 'feattext' then
  --   local src_file = opt.train_src
  --   if dataType == 'monotext' then
  --     src_file = opt.train
  --   end
  --   data.dicts.src = Vocabulary.init('source',
  --                                    src_file,
  --                                    opt.src_vocab or opt.vocab,
  --                                    opt.src_vocab_size or opt.vocab_size,
  --                                    opt.src_words_min_frequency or opt.words_min_frequency,
  --                                    opt.features_vocabs_prefix,
  --                                    function(s) return isValid(s, opt.src_seq_length or opt.seq_length) end,
  --                                    opt.keep_frequency,
  --                                    opt.idx_files)
  -- end
  -- if dataType ~= 'monotext' then
  --   local tgt_file = opt.train_tgt
  --   data.dicts.tgt = Vocabulary.init('target',
  --                                    tgt_file,
  --                                    opt.tgt_vocab,
  --                                    opt.tgt_vocab_size,
  --                                    opt.tgt_words_min_frequency,
  --                                    opt.features_vocabs_prefix,
  --                                    function(s) return isValid(s, opt.tgt_seq_length) end,
  --                                    opt.keep_frequency,
  --                                    opt.idx_files)
  -- end

  -- _G.logger:info('Preparing training data...')

  -- local parallelValidFunc = nil
  -- if opt.check_plength then
  --   parallelValidFunc = parallelCheck
  -- end

  -- data.train = {}
  -- if dataType == 'monotext' then
  --   data.train.src = Preprocessor:makeMonolingualData(opt.train, data.dicts.src, isValid)
  -- elseif dataType == 'feattext' then
  --   data.train.src, data.train.tgt = Preprocessor:makeFeatTextData(opt.train_src, opt.train_tgt,
  --                                                                  data.dicts.tgt,
  --                                                                  isValid, parallelValidFunc)
  --   -- record the size of the input layer
  --   data.dicts.srcInputSize = data.train.src.vectors[1]:size(2)
  -- else
  --   data.train.src, data.train.tgt = Preprocessor:makeBilingualData(opt.train_src, opt.train_tgt,
  --                                                                   data.dicts.src, data.dicts.tgt,
  --                                                                   isValid, parallelValidFunc)
  -- end

  -- _G.logger:info('')

  -- _G.logger:info('Preparing validation data...')
  -- data.valid = {}
  -- if dataType == 'monotext' then
  --   data.valid.src = Preprocessor:makeMonolingualData(opt.valid, data.dicts.src, isValid)
  -- elseif dataType == 'feattext' then
  --   data.valid.src, data.valid.tgt = Preprocessor:makeFeatTextData(opt.valid_src, opt.valid_tgt,
  --                                                                   data.dicts.tgt,
  --                                                                   isValid)
  -- else
  --   data.valid.src, data.valid.tgt = Preprocessor:makeBilingualData(opt.valid_src, opt.valid_tgt,
  --                                                                   data.dicts.src, data.dicts.tgt,
  --                                                                   isValid)
  -- end

  -- _G.logger:info('')

  -- if dataType == 'monotext' then
  --   if opt.vocab:len() == 0 then
  --     Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.dict')
  --   end
  --   if opt.features_vocabs_prefix:len() == 0 then
  --     Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data)
  --   end
  -- elseif dataType == 'feattext' then
  --   if opt.tgt_vocab:len() == 0 then
  --     Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
  --   end
  --   if opt.features_vocabs_prefix:len() == 0 then
  --     Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data)
  --   end
  -- else
  --   if opt.src_vocab:len() == 0 then
  --     Vocabulary.save('source', data.dicts.src.words, opt.save_data .. '.src.dict')
  --   end

  --   if opt.tgt_vocab:len() == 0 then
  --     Vocabulary.save('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
  --   end
  --   if opt.features_vocabs_prefix:len() == 0 then
  --     Vocabulary.saveFeatures('source', data.dicts.src.features, opt.save_data..'.source')
  --     Vocabulary.saveFeatures('target', data.dicts.tgt.features, opt.save_data..'.target')
  --   end
  -- end

  -- _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  -- torch.save(opt.save_data .. '-train.t7', data, 'binary', false)
  -- _G.logger:shutDown()
end

main()
