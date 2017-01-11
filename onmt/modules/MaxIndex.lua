local MaxIndex, parent = torch.class('onmt.MaxIndex', 'nn.Container')

function MaxIndex:__init(dimension, nInputDims)
  parent.__init(self)
  self.net = nn.Max(dimension, nInputDims)
  self:add(self.net)
end

function MaxIndex:updateOutput(input)
  self.net:updateOutput(input)
  self.output:resize(self.net._indices:view(-1):size())
  self.output:copy(self.net._indices:view(-1))
  return self.output
end

function MaxIndex:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end


function MaxIndex:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
