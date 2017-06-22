local tds

local TextPreprocessor, parent = torch.class('TextPreprocessor', 'Preprocessor')

TextPreprocessor.tokensSeparator = ' '
TextPreprocessor.featuresSeparator = '￨'

function TextPreprocessor:__init(dicts, maxLength)
  self.dicts = dicts
  parent.__init(self, maxLength)
end

function TextPreprocessor:consume(file)
  return file:read()
end

function TextPreprocessor:tokenize(line)
  -- Split on delimiter.
  local tokens = onmt.utils.String.split(line, TextPreprocessor.tokensSeparator)

  -- Extract features from words.
  local words = {}
  local features = {}
  local numFeatures = nil

  for t = 1, #tokens do
    local fields = onmt.utils.String.split(tokens[t], TextPreprocessor.featuresSeparator)
    local word = fields[1]

    if word:len() > 0 then
      table.insert(words, word)

      if numFeatures == nil then
        numFeatures = #fields - 1
      else
        assert(#fields - 1 == numFeatures,
               'all words must have the same number of features')
      end

      if #fields > 1 then
        for i = 2, #fields do
          if features[i - 1] == nil then
            features[i - 1] = {}
          end
          table.insert(features[i - 1], fields[i])
        end
      end
    end
  end

  return words, features
end

function TextPreprocessor:detokenize(words, features)
  local tokens

  if not features or #features == 0 then
    tokens = words
  else
    tokens = {}

    for i = 1, #words do
      tokens[i] = words[i]
      for j = 1, #features do
        tokens[i] = tokens[i] .. TextPreprocessor.featuresSeparator .. features[j][i]
      end
    end
  end

  return table.concat(tokens, TextPreprocessor.tokensSeparator)
end

function TextPreprocessor:process(line)
  local words, features = self:tokenize(line)

  local data = {}
  data.length = #words
  data.input = self:processWords(words)
  data.pruned = data.input:eq(onmt.Constants.UNK):sum() / data.length

  if #self.dicts.features > 0 then
    data.meta = self:processFeatures(features)
  end

  return data
end

function TextPreprocessor:generate(wordIds, featuresIds)
  local words = self:generateWords(wordIds)
  local features = self:generateFeatures(featuresIds)
  return self:detokenize(words, features)
end

function TextPreprocessor:processWords(words)
  return self.dicts.words:convertToIdx(words, onmt.Constants.UNK_WORD)
end

function TextPreprocessor:generateWords(wordIds)
  return self.dicts.words:convertToLabels(wordIds, onmt.Constants.EOS)
end

function TextPreprocessor:processFeatures(features)
  local featuresIds = self:_initFeatures(features)

  for j = 1, #self.dicts.features do
    featuresIds[j] = self.dicts.features[j]:convertToIdx(features[j], onmt.Constants.UNK_WORD)
  end

  return featuresIds
end

function TextPreprocessor:generateFeatures(featuresIds)
  local numFeatures = #featuresIds[1]

  if numFeatures == 0 then
    return {}
  end

  local features = {}

  for _ = 1, numFeatures do
    table.insert(features, {})
  end

  for i = 2, #featuresIds do
    for j = 1, numFeatures do
      table.insert(features[j], self.dicts.features[j]:lookup(featuresIds[i][j]))
    end
  end

  return features
end

function TextPreprocessor:_initFeatures(features)
  assert(#self.dicts.features == #features,
         "expected " .. #self.dicts.features .. " features, got " .. #features)

  local success

  if tds == nil then
    success, tds = pcall(require, 'tds')
    if not success then
      tds = false
    end
  end

  return tds and tds.Vec() or {}
end

return TextPreprocessor
