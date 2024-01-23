using DrWatson
@quickactivate "Domain_transfer"
using Random
using Metalhead
using Flux
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM , gpu
using LinearAlgebra
using PyPlot
using MLDatasets
using Statistics
using Images 
using JLD2
using BSON: @save 
using Printf 
using MLUtils
using Random; Random.seed!(1)


model = Chain(
    Conv((3, 3), 1=>64, relu;pad=1),
    x -> maxpool(x, (2,2)),
    Conv((3, 3), 64=>32, relu;pad=1),
    x -> maxpool(x, (2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(512, 1),
    sigmoid
)

model = gpu(model)

# Training hyperparameters 
n_epochs     = 20
# device = gpu
lr = 1f-3
opt = Flux.Optimise.ADAM(lr) |> gpu
loss(x, y) = mean(Flux.binarycrossentropy.(x, y))



#Loading Data
# Load in training data
range_digA = [1]
range_digB = [7]
# load in training data
train_x, train_y = MNIST.traindata()

# grab the digist to train on 
inds = findall(x -> x in range_digA, train_y)
train_yA = train_y[inds]
train_xA = zeros(Float32,16,16,1,5923)
for i=1:5851
  train_xA[:,:,:,i] = reverse(imresize(train_x[:,:,inds[i]],(16,16)),dims=1)
  train_xA[:,:,1,i] = rotr90(train_xA[:,:,1,i],1)
  # train_xA[:,:,:,i] = train_x[:,:,inds[i]]
end

inds = findall(x -> x in range_digB, train_y)
train_yB = train_y[inds]
train_xB = zeros(Float32,16,16,1,5851)
for i=1:5851

  train_xB[:,:,:,i] = reverse(imresize(train_x[:,:,inds[i]],(16,16)),dims=1)
  train_xB[:,:,1,i] = rotr90(train_xB[:,:,1,i],1)
  # train_xB[:,:,:,i] = train_x[:,:,inds[i]] 
end
train_x = cat(train_xA, train_xB,dims=4)
train_y = cat(zeros(5851), ones(5851),dims=1)
num_classes = 2

# Perform one-hot encoding using Flux.onehotbatch
# train_x = normalize_images(train_x)
# test_x = normalize_images(test_x)
idx = reshape(randperm(11702), 11702, 1)

train_x = train_x[:,:,:,idx]
train_y = train_y[idx]
test_x = train_x[:,:,:,10048:10560]
train_x = train_x[:,:,:,1:10048]

test_y = train_y[10048:10560]
train_y = train_y[1:10048]

n_train = size(train_x)[end]
n_test = size(test_x)[end]-1
batch_size=64
n_batches = cld(n_train, batch_size) 
n_batches_test = cld(n_test, batch_size) 
# Training Loop
losslist=[]
accuracylist=[]

losslistpara=[]
accuracylistpara=[]

losslist_test=[]
accuracylist_test=[]
for e=1:n_epochs
  epoch_loss=0.0
  epoch_loss_test=0.0
  correct_predictions=0.0
  correct_predictions_test=0.0
  idx_e = reshape(randperm(n_train), batch_size, n_batches)
    for b = 1:n_batches # batch loop
        x = train_x[:,:,:,idx_e[:,b]]
        x .+= 0.001*randn(Float32, size(x))
        x = x |> gpu
        y = transpose(train_y[idx_e[:,b]])|>gpu
        grads = Flux.gradient(Flux.params(model)) do
          y_pred = model(x)
          l = loss(y_pred,y)
        end
        # epoch_loss +=Loss
        Flux.update!(opt,Flux.params(model),grads)
        
        
        print("batch: $b, epoch: $e \n ")
        # Calculate accuracy for this batch
        y_pred = model(x)
        epoch_loss += loss(y_pred,y)
        correct_predictions += sum(round.(y_pred) .== y)
    end

    idx_e_test = reshape(randperm(n_test), batch_size, n_batches_test)  
    for b = 1:n_batches_test # batch loop
      x = test_x[:,:,:,idx_e_test[:,b]]
      x .+= 0.001*randn(Float32, size(x))
      x = x |> gpu
      y = transpose(test_y[idx_e_test[:,b]])|>gpu
      
      print("batch: $b \n")
      # Calculate accuracy for this batch
      y_pred = model(x)
      epoch_loss_test += loss(y_pred,y)
      correct_predictions_test += sum(round.(y_pred) .== y)
  end
    
    # Calculate average loss for the epoch
    avg_epoch_loss_test = epoch_loss_test / size(idx_e_test, 2)
    accuracy_test = correct_predictions_test / 512

    push!(losslist_test, avg_epoch_loss_test)
    push!(accuracylist_test, accuracy_test)

    avg_epoch_loss = epoch_loss_test / size(idx_e, 2)
    accuracy = correct_predictions / 10048
    push!(losslist, avg_epoch_loss)
    push!(accuracylist, accuracy)
    
    println("Epoch $e, Average Loss: $avg_epoch_loss, Accuracy: $accuracy% , Average Loss test: $avg_epoch_loss_test, Accuracy test: $accuracy_test%\n")

    plt.plot(1:e,losslist[1:e],label="train")
    plt.plot(1:e,losslist_test[1:e],label="test")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("Loss per epoch")
    plt.savefig("../plots/loss $e.png")
    plt.close()


    plt.plot(1:e,accuracylist[1:e],label="train")
    plt.plot(1:e,accuracylist_test[1:e],label="test")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.title("acc per epoch")
    plt.savefig("../plots/acc $e.png")
    plt.close()
end

plt.plot(losslist,label="train")
plt.plot(losslist_test,label="test")
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.title("Loss per epoch")
plt.savefig("../plots/loss.png")
plt.close()


plt.plot(accuracylist,label="train")
plt.plot(accuracylist_test,label="test")
plt.xlabel("epoch")
plt.ylabel("value")
plt.legend()
plt.title("acc per epoch")
plt.savefig("../plots/acc.png")
plt.close()