local Dict = require('onmt.utils.Dict')
local Constants = require('onmt.Constants')
local Table = require('onmt.utils.Table')

--[[ Class to build vocabularies, containing one or several dictionaries. ]]
local Vocabulary = torch.class('Vocabulary')

--[[ Static function to build vocabulary related options as needed.

Parameters:

  * `prefix` - an optional option prefix (e.g. `src`).

Returns: the table of options.

]]
function Vocabulary.addOpts(options, prefix, name)
  prefix = prefix and prefix .. '_' or ''

  Table.append(
    options,
    {
      {
        '-' .. prefix ..'vocab', { },
        [[List of pre-built vocabularies: `words.txt[ feat1.txt[ feat2.txt[ ...] ] ]`.
        Use 'x' if you don't want to set a pre-built vocabulary for a stream.]]
      },
      {
        '-' .. prefix .. 'vocab_size', { 50000 },
        [[List of vocabularies size: `word[ feat1[ feat2[ ...] ] ]`.
        If = 0, vocabularies are not pruned.]]
      },
      {
        '-' .. prefix .. 'words_min_frequency', { 0 },
        [[List of words minimum frequency: `word[ feat1[ feat2[ ...] ] ]`.
        If = 0, vocabularies are pruned by size.]]
      }
    }
  )

  return options
end

--[[ Creates a new vocabulary.

Parameters:

  * `args` - a table containing the keys as listed in `addOpts`.
  * `fileIterator` - a text file iterator.
  * `splitter` - a `DataTransformer` to segment the text.
  * `prefix` - the prefix used to build the options.

]]
function Vocabulary:__init(args, fileIterator, splitter, prefix)
  self.dicts = {}

  -- Read prefixed options.
  prefix = prefix and prefix .. '_' or ''
  local vocabOpt = args[prefix .. 'vocab']
  local vocabSizeOpt = args[prefix .. 'vocab_size']
  local minFrequencyOpt = args[prefix .. 'words_min_frequency']

  self.prefix = prefix

  -- Lookup first line to figure the number of streams.
  local numStreams = #splitter:transform(fileIterator:lookup())

  -- Load pre-buillt vocabulary, if any.
  local numLoaded = self:_loadFromFiles(vocabOpt)

  -- If some stream vocabularies are missing, we need to process the text file.
  if numLoaded ~= numStreams then
    local dicts = self:_buildFromText(fileIterator, splitter)

    for i = 1, numStreams do
      if not self.dicts[i] then
        -- Also pruned generated vocabulary as needed.
        local maxSize = vocabSizeOpt[i] or 0
        local minFrequency = minFrequencyOpt[i] or 0

        if minFrequency > 0 then
          self.dicts[i] = dicts[i]:pruneByMinFrequency(minFrequency)
        elseif maxSize > 0 then
          self.dicts[i] = dicts[i]:prune(maxSize)
        else
          self.dicts[i] = dicts
        end
      end
    end

  end

end

--[[ Saves dictionaries on disk.

Parameters:

  * `path` - the prefix of the path.

]]
function Vocabulary:save(path)
  for i = 1, #self.dicts do
    local file = path
    if self.prefix ~= '' then
      file = file .. '.' .. self.prefix
    end
    file = file .. '.dict.' .. tostring(i)

    self.dicts[i]:writeFile(file)
  end
end

--[[ Builds dictionaries from a text file. ]]
function Vocabulary:_buildFromText(fileIterator, splitter)
  local dicts = {}

  while not fileIterator:isEOF() do
    local item = fileIterator:next()
    local streams = splitter:transform(item)

    if #dicts == 0 then
      for i = 1, #streams do
        dicts[i] = Dict.new({Constants.PAD_WORD, Constants.UNK_WORD,
                             Constants.BOS_WORD, Constants.EOS_WORD})
      end
    else
      assert(#streams == #dicts, 'all sentences must have the same number of streams')
    end

    for i = 1, #streams do
      for j = 1, #streams[i] do
        dicts[i]:add(streams[i][j])
      end
    end
  end

  return dicts
end

--[[ Loads pre-built dictionaries. ]]
function Vocabulary:_loadFromFiles(vocabs)
  local loaded = 0

  for i = 1, #vocabs do
    if vocabs[i] ~= 'x' then
      self.dicts[i] = Dict.new(vocabs[i])
      loaded = loaded + 1
    end
  end

  return loaded
end

return Vocabulary
