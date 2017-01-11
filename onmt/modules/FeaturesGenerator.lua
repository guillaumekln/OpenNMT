--[[ Feature decoder generator. Given RNN state, produce categorical distribution over
tokens and features.

  Implements $$[softmax(W^1 h + b^1), softmax(W^2 h + b^2), ..., softmax(W^n h + b^n)] $$.
--]]


local FeaturesGenerator, parent = torch.class('onmt.FeaturesGenerator', 'nn.Container')

--[[
Parameters:

  * `rnnSize` - Input rnn size.
  * `outputSize` - Output size (number of tokens).
  * `features` - table of feature sizes.
--]]
function FeaturesGenerator:__init(rnnSize, outputSize, targetLookup, features)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize, targetLookup, features)
  self:add(self.net)
end

function FeaturesGenerator:_buildGenerator(rnnSize, outputSize, targetLookup, features)
  local outputs = {}

  local decOut = nn.Identity()()

  local wordGen = nn.SelectTable(1)(onmt.Generator(rnnSize, outputSize)(decOut))
  table.insert(outputs, wordGen)

  local bestWord = onmt.MaxIndex(1, 1)(wordGen)
  local bestWordEmb = targetLookup(bestWord)

  local context = nn.JoinTable(2)({decOut, bestWordEmb})

  for i = 1, #features do
    local map = nn.Linear(rnnSize + targetLookup.vecSize, features[i]:size())(context)
    local loglk = nn.LogSoftMax()(map)
    table.insert(outputs, loglk)
  end

  return nn.gModule({decOut}, outputs)
end

function FeaturesGenerator:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function FeaturesGenerator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function FeaturesGenerator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
