require('onmt.init')

local InMemoryDataset = require('onmt.data.InMemoryDataset')
local FeatToText = require('onmt.data.providers.FeatToText')
local TextToText = require('onmt.data.providers.TextToText')
local MonotText = require('onmt.data.providers.MonoText')

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
    '-report_every', 100000,
    [[Report status every this many entries.]],
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

local cmd = ExtendedCmdLine.new('preprocess.lua')

local dataType = cmd.getArgument(arg, '-data_type') or 'bitext'
local dataClass

if dataType == 'bitext' then
  dataClass = TextToText
elseif dataType == 'feattext' then
  dataClass = FeatToText
elseif dataType == 'monotext' then
  dataClass = MonoText
end

cmd:setCmdLineOptions(options, 'Preprocess')
dataClass.declareOpts(cmd)
cmd:setCmdLineOptions(otherOptions, 'Other')
Logger.declareOpts(cmd)

local function main()
  local opt = cmd:parse(arg)

  torch.manualSeed(opt.seed)

  _G.logger = Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local dataManager = dataClass.new(opt)

  local function buildDataset(iterators, vocabs)
    local dataset = InMemoryDataset.new()

    dataset = dataset:map(dataManager:getMappers(vocabs))
    dataset = dataset:setValidators(dataManager:getValidators())
    dataset = dataset:setParallelValidator(dataManager:getParallelValidator())

    dataset = dataset:fromFiles(iterators, opt.report_every)

    if opt.shuffle then
      _G.logger:info('... shuffling entries')
      dataset = dataset:shuffle()
    end

    if opt.sort then
      _G.logger:info('... sorting entries')
      dataset = dataset:sort(dataManager:sortFunc())
    end

    _G.logger:info('Prepared %d entries', dataset:size())
    _G.logger:info(' * %d discarded for failing validation', dataset.discarded)

    for i = 1, #iterators do
      iterators[i]:close()
    end

    return dataset
  end

  local trainIterators = dataManager:getTrainIterators()

  local vocabs = dataManager:getVocabularies(trainIterators)
  _G.logger:info('')

  _G.logger:info('Preparing training data...')
  local trainDataset = buildDataset(trainIterators, vocabs)
  _G.logger:info('')

  local validIterators = dataManager:getValidIterators()

  _G.logger:info('Preparing validation data...')
  local validDataset = buildDataset(validIterators, vocabs)
  _G.logger:info('')

  for i = 1, #vocabs do
    vocabs[i]:save(opt.save_data)
  end

  _G.logger:info('')
  _G.logger:info('Saving data to \'' .. opt.save_data .. '-train.t7\'...')

  local data = {}
  data.vocabs = vocabs
  data.train = trainDataset.data
  data.valid = validDataset.data
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)

  _G.logger:shutDown()
end

main()
