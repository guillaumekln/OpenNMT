local TextProcessor, parent = torch.class('TextProcessor', 'Processor')

TextProcessor.tokensSeparator = ' '
TextProcessor.featuresSeparator = '￨'

function TextProcessor:__init(dicts, maxLength)
  self.dicts = dicts
  parent.__init(self, maxLength)
end

function TextProcessor:consume(file)
  return file:read()
end

function TextProcessor:tokenize(line)
  -- Split on delimiter.
  local tokens = onmt.utils.String.split(line, TextProcessor.tokensSeparator)

  -- Extract features from words.
  local words = {}
  local features = {}
  local numFeatures = nil

  for t = 1, #tokens do
    local fields = onmt.utils.String.split(tokens[t], TextProcessor.featuresSeparator)
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

function TextProcessor:detokenize(words, features)
  local tokens

  if not features or #features == 0 then
    tokens = words
  else
    tokens = {}

    for i = 1, #words do
      tokens[i] = words[i]
      for j = 1, #features do
        tokens[i] = tokens[i] .. TextProcessor.featuresSeparator .. features[j][i]
      end
    end
  end

  return table.concat(tokens, TextProcessor.tokensSeparator)
end

function TextProcessor:process(line)
  local words, features = self:tokenize(line)

  local data = {}
  data.length = #words
  data.input = self:processWords(words)
  data.meta = self:processFeatures(features)
  data.pruned = data.input:eq(onmt.Constants.UNK):sum() / data.length
  return data
end

function TextProcessor:generate(wordIds, featuresIds)
  local words = self:generateWords(wordIds)
  local features = self:generateFeatures(featuresIds)
  return self:detokenize(words, features)
end

function TextProcessor:processWords(words)
  return self.dicts.words:convertToIdx(words, onmt.Constants.UNK_WORD)
end

function TextProcessor:generateWords(wordIds)
  return self.dicts.words:convertToLabels(wordIds, onmt.Constants.EOS)
end

function TextProcessor:processFeatures(features)
  local featuresIds = {}

  for j = 1, #self.dicts.features do
    featuresIds[j] = self.dicts.features[j]:convertToIdx(features[j], onmt.Constants.UNK_WORD)
  end

  return featuresIds
end

function TextProcessor:generateFeatures(featuresIds)
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

return TextProcessor
