--[[ Base class for providers: classes that describe how to iterate, transform and validate data.

See `preprocess.lua` for how it plays with a Dataset.

]]
local Provider = torch.class('Provider')

function Provider:__init()
end

function Provider:getTrainIterators()
  error('Not implemented')
end

function Provider:getValidIterators()
  error('Not implemented')
end

function Provider:getVocabularies(_)
  return {}
end

function Provider:getMappers(_)
  return {}
end

function Provider:getParallelValidator()
  return nil
end

function Provider:getValidators()
  return nil
end

function Provider:sortFunc()
  return 0
end

return Provider
