--[[ Base class for data transformers: classes that transform data from one state
  to another and optionally the other way.
]]
local DataTransformer = torch.class('DataTransformer')

function DataTransformer:__init()
end

function DataTransformer:tranform(_)
  error('Not implemented')
end

function DataTransformer:reverse(_)
  error('Not implemented')
end

return DataTransformer
