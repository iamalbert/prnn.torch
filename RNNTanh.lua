local RNNTanh, parent = torch.class('prnn.RNNTanh', 'prnn.RNN')

function RNNTanh:__init(inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    parent.__init(self,inputSize, hiddenSize, numLayers, batchFirst, dropout, rememberStates)
    self.mode = 'PRNN_RNN_TANH'
    self:reset()
end
