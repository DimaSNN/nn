

-- рабочая сеть для CPU
require 'torch'
require 'nn'
require 'csvigo'
require 'nnx'
require 'optim'



--trainPath = '/home/dmitry/nn/Spectr_MFNN.csv'
trainPath = '/home/dmitry/nn/data/Спектр.csv'

trainData= csvigo.load({path= trainPath, mode='raw', separator=';' })
-- testData = torch.load(testPath,'ascii')
csv_tensor = torch.Tensor(trainData)



input= csv_tensor:sub(1,9,1,3)
output1 = csv_tensor:sub(1,9,4,15004)

N= output1:size(2) --входное число в спектре
n= 1024 --выходное число нейронов 

--изменение размерности
set = (N/n)

--output=torch.Tensor(9,1024)	
--output=nil
for i=0, n-1,1 do
	 local from= math.ceil(i *set)+1
	 local to=math.ceil((i+1)*set)
	if i>0 then
		output=torch.cat(output, torch.sum(output1:sub(1,9,from,to), 2)) -- /(1+to-from) --/(1+to-from)
	else
		--oo=nil
		output=torch.sum(output1:sub(1,9,from,to), 2) --/(to-from+1)
	end	
end

output=torch.Tensor(output)


-- define the mlp
mlp = nn.Sequential()

mlp:add(nn.Reshape(input:size(2)))
mlp:add(nn.Linear(input:size(2), 12))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(12,128))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(128, 512))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(512, output:size(2)))


loss = nn.MSECriterion() 


optim_state_lbfgs = {
	learningRate = 0.5,
	maxIter = 60,
	nCorrection = 20
}


w,dE_dw = mlp:getParameters()

function eval_E(w)
--local eval_E = function(w)
    dE_dw:zero() -- Обновляем градиенты
    local Y = mlp:forward(input)
    local E = loss:forward(Y,output)
    local dE_dy = loss:backward(Y,output)
    mlp:backward(input,dE_dy)
    return E, dE_dw
	--return E, dE_dy --для пробы, возможно неправильно
end
-- optim_method = optim.sgd



--закомментить
--local eval_E = function(w)
--    dE_dw:zero() -- Обновляем градиенты
--    local Y = mlp:forward(input)
--    local E = loss:forward(Y,output)
--    local dE_dy = loss:backward(Y,output)
--    mlp:backward(input,dE_dy)
--   return E, dE_dw
--	--return E, dE_dy --для пробы, возможно неправильно
--end
mlp:reset()
timer = torch.Timer();
--maxIter =50;
--for i=1,maxIter do
	_,fw = optim.lbfgs(eval_E, w, optim_state_lbfgs);
--	if i%(torch.floor(maxIter/10))==0 then print(string.format('MSE = %f',fw[1])) end
--end
--cutorch.synchronize()
print(string.format('Success! Average iteration time was %f',timer:time().real))
print('Success!')
#fw
--torch.cat(mlp:forward(input),output)
