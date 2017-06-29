--- GPU---------------------------------------------------------
require 'torch'
require 'nn'
require 'csvigo'
require 'nnx'
require 'optim'
require 'cutorch'
require 'cunn'



--trainPath = '/home/dmitry/nn/Spectr_MFNN.csv'
trainPath = '/home/dmitry/nn/data/Спектр.csv'

trainData= csvigo.load({path= trainPath, mode='raw', separator=';' })
-- testData = torch.load(testPath,'ascii')
csv_tensor = torch.Tensor(trainData)



input= csv_tensor:sub(1,9,1,3)
input = input:cuda()
output1 = csv_tensor:sub(1,9,4,15004)

N= output1:size(2) --входное число в спектре
n= 1024 --выходное число нейронов 

--изменение размерности
set = (N/n)

--output=torch.Tensor(9,1024)	

function GetOutputs (oo)
	local out
	for i=0, n-1,1 do
		 local from= math.ceil(i *set)+1
		 local to=math.ceil((i+1)*set)
		if i>0 then
			out=torch.cat(out, torch.sum(oo:sub(1,9,from,to)/(to-from+1), 2))
		else
			out= torch.sum(oo:sub(1,9,from,to), 2)/(to-from+1)
		end	
	end
	return out
end

output = GetOutputs(output1)
output=output:cuda()
--
--input= csv_tensor:sub(1,30,2,1667)
--input = input:cuda()
--output = csv_tensor:sub(1,30,1,1)
--output=output:cuda()


-- define the mlp
mlp = nn.Sequential()

mlp:add(nn.Reshape(input:size(2)))
mlp:add(nn.Linear(input:size(2), 24))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(24,128))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(128, 512))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(512, n))
mlp:cuda()



loss = nn.MSECriterion() 
--loss = nn.ClassNLLCriterion()

loss:cuda()
--mlp:add(nn.LogSoftMax())
--loss = nn.ClassNLLCriterion()

-- Configuring optimizer
local optim_state = {
   learningRate = 0.01,
   momentum = 0.6,--0.1,
   weightDecay = 0.00005 --1e-6
}
--lbfgs
optim_state_lbfgs = {
	learningRate = 1,
	maxIter = 1000,
	nCorrection = 100
}
config = {
	maxIter=100,

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
fw=nil
timer = torch.Timer();
--for i= 1,10 do

	--_,fw = optim.lbfgs(cuda_eval, w, optim_state_lbfgs);
	_,fw = optim.sg(cuda_eval, w,config);

--end
cutorch.synchronize()
print(string.format('Success! Average iteration time was %f', timer:time().real))
print('Success!')

#fw
--torch.cat(mlp:forward(input[1]), output[1],2)

torch.cat(mlp:forward(input[1])[{{1,40}}], output[1][{{1,40}}],2)

