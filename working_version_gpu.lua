

--- конец
--- GPU---------------------------------------------------------
require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'cutorch'
require 'cunn'
-- require 'csvigo'
csvfile = require "simplecsv"



trainPath = '/home/dmitry/nn/Spectr_MFNN.csv'
--trainPath = '/home/dmitry/nn/data/Спектр.csv'

td = csvfile.read(trainPath, ';')
-- testData = torch.load(testPath,'ascii')
csv_tensor = torch.Tensor(td)



--input= csv_tensor:sub(1,9,1,3)
--input = input:cuda()
--output = csv_tensor:sub(1,9,4,15004)
--output=output:cuda()
--
input= csv_tensor:sub(1,30,2,1667)
input = input:cuda()
output = csv_tensor:sub(1,30,1,1)
output=output:cuda()

-- define the mlp
mlp = nn.Sequential()

mlp:add(nn.Reshape(1666))
mlp:add(nn.Linear(1666, 512))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(512, 256))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(256, 128))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(128, 1))
mlp:cuda()

loss = nn.MSECriterion()
loss:cuda()
--mlp:add(nn.LogSoftMax())
--loss = nn.ClassNLLCriterion()

-- Configuring optimizer
--local optim_state = {
--   learningRate = 0.01,
--   momentum = 0.6,--0.1,
--   weightDecay = 0.0005--1e-5
--}
--lbfgs
optim_state_lbfgs = {
	learningRate = 0.5,
	maxIter = 60,
	nCorrection = 20
}

w,dE_dw = mlp:getParameters()
--dE_dw:cuda()
print("Starting gradient descent from 'optim' on GPU...")
function  cuda_eval(w)
    dE_dw:zero() -- Обновляем градиенты
    local Y = mlp:forward(input)
    local E = loss:forward(Y,output)
    local dE_dy = loss:backward(Y,output)
    mlp:backward(input,dE_dy)
    return E, dE_dw
	--return E, dE_dy --для пробы, возможно неправильно
end

mlp:reset()
timer = torch.Timer();
	_,fw = optim.lbfgs(cuda_eval, w, optim_state_lbfgs);

cutorch.synchronize()
print(string.format('Success! Average iteration time was %f', timer:time().real))
print('Success!')

torch.cat(mlp:forward(input),output)
#fw
print('time=',timer:time().real)
--timer = nil
--конец



