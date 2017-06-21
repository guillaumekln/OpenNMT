local TargetTextPreprocessor, parent = torch.class('TargetTextPreprocessor', 'TextPreprocessor')

function TargetTextPreprocessor:__init(dicts, maxLength, shiftFeatures)
  parent.__init(self, dicts, maxLength)
  self.shiftFeatures = shiftFeatures
end

function TargetTextPreprocessor:convertWords(words)
  return self.dicts.words:convertToIdx(words,
                                       onmt.Constants.UNK_WORD,
                                       onmt.Constants.BOS_WORD,
                                       onmt.Constants.EOS_WORD)
end

function TargetTextPreprocessor:convertFeatures(features)
  local featuresIds = self:_initFeatures(features)

  for j = 1, #self.dicts.features do
    -- if shift_feature then target features are shifted relative to the target words.
    -- Use EOS tokens as a placeholder.
    table.insert(features[j], 1, onmt.Constants.BOS_WORD)
    if self.shiftFeatures then
      table.insert(features[j], 1, onmt.Constants.EOS_WORD)
    else
      table.insert(features[j], onmt.Constants.EOS_WORD)
    end

    featuresIds[j] = self.dicts.features[j]:convertToIdx(features[j], onmt.Constants.UNK_WORD)

    table.remove(features[j], 1)
    if self.shiftFeatures then
      table.remove(features[j], 1)
    else
      table.remove(features[j])
    end
  end

  return featuresIds
end

return TargetTextPreprocessor
