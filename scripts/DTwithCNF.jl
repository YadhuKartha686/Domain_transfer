using DrWatson
@quickactivate "Domain_transfer"

using Flux
using LinearAlgebra
using Random
using Metalhead
using JLD2
using Statistics
using ImageQualityIndexes 
using PyPlot
using SlimPlotting
using InvertibleNetworks
using MLUtils
using BSON 
# using Wavelets
using StatsBase
using Distributions
using Images

#### DATA LOADING #####
nx,ny = 1024, 256
N = nx*ny;

data_path= "../data/CompassShot.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
train_xA = zeros(Float32, nx, ny, 1,900)
train_xB = zeros(Float32, nx, ny, 1,900)


indices_of_A = findall(x -> x == 0.0, train_y[:,1])
indices_of_B = findall(x -> x == 1.0, train_y[:,1])
  
for i=1:900
    sigma = 1.0
    train_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_A[i]],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
    train_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_B[i]],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
end



# Define the generator and discriminator networks

device = gpu #GPU does not accelerate at this small size. quicker on cpu
lr     = 1f-5
epochs = 30
batch_size = 1
low = 0.5f0

# Architecture parametrs
chan_x = 1; chan_y = 1; L = 4; K = 10; n_hidden = 128 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K; split_scales=true,activation=SigmoidLayer(low=low,high=1.0f0)) |> device;

model = Chain(
  # First convolutional layer
  Conv((7, 7), 1=>32, relu, pad=(3,3), stride=(2,2)),
  MaxPool((3,3), stride=(2,2)), # Aggressive pooling to reduce dimensions
  
  # Second convolutional layer
  Conv((5, 5), 32=>64, relu, pad=(2,2), stride=(2,2)),
  MaxPool((3,3), stride=(2,2)), # Further reduction
  
  # Third convolutional layer
  Conv((5, 5), 64=>64, relu, pad=1),
  MaxPool((2,2), stride=(2,2)), # Final pooling to reduce to a very low dimension
  # Fourth convolutional layer
  Conv((3, 3), 64=>64, relu, pad=1),
  MaxPool((2,2), stride=(2,2)), # Final pooling to reduce to a very low dimension

  Conv((3, 3), 64=>64, relu, pad=1),
  MaxPool((2,2), stride=(2,2)), # Final pooling to reduce to a very low dimension
  
  # Flatten the output of the last convolutional layer before passing it to the dense layer
  Flux.flatten,
  
  # Fully connected layer
  Dense(448, 128, relu),
  
  # Output layer for binary classification
  Dense(128, 1),
  sigmoid
)



# Define the loss functions
function Dissloss(real_output, fake_output)
  real_loss = mean(Flux.binarycrossentropy.(real_output, 1f0))
  fake_loss = mean(Flux.binarycrossentropy.(fake_output, 0f0))
  return 0.5f0*(real_loss + fake_loss)
end

function Genloss(fake_output,x,y) 
  return mean(Flux.binarycrossentropy.(fake_output, 1f0)) + 0*Flux.mse(y|> device,x)
end


# Initialize networks and optimizers
generator = G |> gpu
discriminatorA = gpu(model)
discriminatorB = gpu(model)

# generator = G
# discriminatorA = model
# discriminatorB = model

opt_adam = "adam"
clipnorm_val = 5f0
optimizer_g = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))
lrd = 1f-5
optimizer_da = Flux.ADAM(lrd)
optimizer_db = Flux.ADAM(lrd)
genloss=[]
dissloss = []
mseofimb=[]
mseofima=[]
imgs = 4
n_train = 800
n_test = 805
n_batches = cld(n_train,imgs)
YA = ones(Float32,nx,ny,1,imgs) + randn(Float32,nx,ny,1,imgs) ./1000
YB = ones(Float32,nx,ny,1,imgs) .*7 + randn(Float32,nx,ny,1,imgs) ./1000

lossnrm      = []; logdet_train = []; 
factor = 1f-17

n_epochs     = 300
for e=1:n_epochs# epoch loop
  epoch_loss_diss=0.0
  epoch_loss_gen=0.0
  idx_eA = reshape(randperm(n_train), imgs, n_batches)
  idx_eB = reshape(randperm(n_train), imgs, n_batches)
  for b = 1:n_batches # batch loop
        @time begin
          ############# Loading domain A data ############## 
          
          XA = train_xA[:, :, :, idx_eA[:,b]];
          XB = train_xB[:, :, :, idx_eA[:,b]];  
          X = cat(XA, XB,dims=4)
          Y = cat(YA, YB,dims=4)

          Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####

          ######## interchanging conditions to get domain transferred images during inverse call #########

          ZyA = Zy[:,:,:,1:imgs]
          ZyB = Zy[:,:,:,imgs+1:end]

          ZxA = Zx[:,:,:,1:imgs]
          ZxB = Zx[:,:,:,imgs+1:end]

          Zy1 = cat(ZyB,ZyA,dims=4)


          fake_images,invcall = generator.inverse(Zx|>device,Zy1)  ###### generating images #######
 
          ####### getting fake images from respective domain ########

          fake_imagesAfromB = fake_images[:,:,:,imgs+1:end]
          fake_imagesBfromA = fake_images[:,:,:,1:imgs]

          ####### discrim training ########
    
            dA_grads = Flux.gradient(Flux.params(discriminatorA)) do
                real_outputA = discriminatorA(XA|> device)
                fake_outputA = discriminatorA(fake_imagesAfromB|> device)
                lossA = Dissloss(real_outputA, fake_outputA)
            end
            Flux.Optimise.update!(optimizer_da, Flux.params(discriminatorA),dA_grads)  #### domain A discrim ####

            
            dB_grads = Flux.gradient(Flux.params(discriminatorB)) do
                real_outputB = discriminatorB(XB|> device)
                fake_outputB = discriminatorB(fake_imagesBfromA|> device)
                lossB = Dissloss(real_outputB, fake_outputB)
            end
            Flux.Optimise.update!(optimizer_db, Flux.params(discriminatorB),dB_grads)  #### domain B discrim ####
        
          ## minlog (1-D(fakeimg)) <--> max log(D(fake)) + norm(Z)
                    
          gsA = gradient(x -> Genloss(discriminatorA(x|> device),x,XA), fake_imagesAfromB)[1]  #### getting gradients wrt A fake ####
          gsB = gradient(x -> Genloss(discriminatorB(x|> device),x,XB), fake_imagesBfromA)[1]  #### getting gradients wrt B fake ####
          

          gs = cat(gsB,gsA,dims=4)
          generator.backward_inv(((gs ./ factor)|>device), fake_images, invcall;) #### updating grads wrt image ####
          generator.backward(Zx / imgs*2, Zx, Zy;)
          # generator.backward_inv(((gsA ./ factor)|>device) + ZxA/4, fake_imagesAfromB, invcall[:,:,:,5:8];) #### updating grads wrt A ####
          # generator.backward_inv(((gsB ./ factor)|>device) + ZxB/4, fake_imagesBfromA, invcall[:,:,:,1:4];) #### updating grads wrt B ####

          for p in get_params(generator)
              Flux.update!(optimizer_g,p.data,p.grad)
          end
          clear_grad!(generator) #### updating generator ####

          #loss calculation for printing
          fake_outputA = discriminatorA(fake_imagesAfromB|> device)
          fake_outputB = discriminatorB(fake_imagesBfromA|> device)
          real_outputA = discriminatorA(XA|> device)
          real_outputB = discriminatorB(XB|> device)

          lossAd = Dissloss(real_outputA, fake_outputA)  #### log(D(real)) + log(1 - D(fake)) ####
          lossBd = Dissloss(real_outputB, fake_outputB)  #### log(D(real)) + log(1 - D(fake)) ####
          lossA = Genloss(fake_outputA,fake_imagesAfromB,fake_imagesAfromB)  #### log(1 - D(fake)) + mse ####
          lossB = Genloss(fake_outputB,fake_imagesAfromB,fake_imagesAfromB)  #### log(1 - D(fake)) + mse ####
          f_all = norm(Zx)^2

          loss = lossA + lossB #+ ml

          append!(lossnrm, f_all / (imgs*2*N))  # normalize by image size and batch size
          append!(logdet_train, (-lgdet) / N) # logdet is internally normalized by batch size
          append!(genloss, loss)  # normalize by image size and batch size
          append!(dissloss, (lossAd+lossBd) ) # logdet is internally normalized by batch size


          epoch_loss_diss += (lossAd+lossBd)
          epoch_loss_gen += loss

          println("Iter: epoch=", e, "/", n_epochs,":batch = ",b,
          "; Genloss=", loss, 
            "; genloss = ",  loss+(f_all / (imgs*2*N)), 
              "; dissloss = ", (lossAd+lossBd)/2 , 
              "; f l2 = ",  lossnrm[end], 
              "; lgdet = ", logdet_train[end], "\n")

          Base.flush(Base.stdout)
        end
        
        if mod(e,1)==0 && mod(b,n_batches)==0
          avg_epoch_lossd = epoch_loss_diss / size(idx_eA, 2)
          avg_epoch_lossg= epoch_loss_gen / size(idx_eA, 2)
          push!(genloss, avg_epoch_lossg)
          push!(dissloss, avg_epoch_lossd)
        end

        if mod(e,5)==0 && mod(b,n_batches)==0
          x = train_xA[:,:,:,801:end]
          x .+= 0.001*randn(Float32, size(x))
          x = x |> gpu
          y = transpose(ones(Float32,100))|>gpu
          # Calculate accuracy for this batch
          y_pred = discriminatorA(x)
          correct_predictions_testa = sum(abs.(y_pred-y))/100

          println("l2norm of Da: ",correct_predictions_testa)

          x = train_xB[:,:,:,801:end]
          x .+= 0.001*randn(Float32, size(x))
          x = x |> gpu
          y = transpose(ones(Float32,100))|>gpu
          # Calculate accuracy for this batch
          y_pred = discriminatorB(x)
          correct_predictions_testb = sum(abs.(y_pred-y))/100

          println("l2norm of Db: ",correct_predictions_testb)
       end

        if mod(e,10) == 0 && mod(b,n_batches)==0
          plt.plot(lossnrm)
          plt.title("loss $e")
          plt.savefig("../plots/Shot_rec_df/lossnorm$e.png")
          plt.close()
          plt.plot(logdet_train)
          plt.title("logdet $e")
          plt.savefig("../plots/Shot_rec_df/logdet$e.png")
          plt.close()
      
          plt.plot(1:e,dissloss[1:e],label = "Diss")
          plt.plot(1:e,genloss[1:e], label = "Gen")
          plt.title("ganloss $e")
          plt.legend()
          plt.savefig("../plots/Shot_rec_df/ganloss$e.png")
          plt.close()
        end

    end
    XA = zeros(Float32 , nx,ny,1,imgs)
    XB = zeros(Float32 , nx,ny,1,imgs)
    XA[:,:,:,1:imgs] = train_xA[:,:,:,n_test:n_test-1+imgs]
    XB[:,:,:,1:imgs] = train_xB[:,:,:,n_test:n_test-1+imgs]

    X = cat(XA, XB,dims=4)
    Y = cat(YA, YB,dims=4)
    Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####


              ######## interchanging conditions to get domain transferred images during inverse call #########

    ZyA = Zy[:,:,:,1:imgs]
    ZyB = Zy[:,:,:,imgs+1:end]

    Zy = cat(ZyB,ZyA,dims=4)

    fake_images,invcall = generator.inverse(Zx|>device,Zy)  ###### generating images #######

              ####### getting fake images from respective domain ########

    fake_imagesAfromBt = fake_images[:,:,:,imgs+1:end]
    fake_imagesBfromAt = fake_images[:,:,:,1:imgs]

    plot_sdata(XB[:,:,1,1]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test1.png")
    plt.close()

    plot_sdata(fake_imagesBfromAt[:,:,1,1]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel+den from vel 1_$e ")
    plt.savefig("../plots/Shot_rec_df/vel+den test pred1_$e.png")
    plt.close()

    plot_sdata(XA[:,:,1,1]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel ")
    plt.savefig("../plots/Shot_rec_df/vel data test1.png")
    plt.close()

    plot_sdata(fake_imagesAfromBt[:,:,1,1]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel from vel+den 1_$e ")
    plt.savefig("../plots/Shot_rec_df/vel test pred1_$e.png")
    plt.close()



    plot_sdata(XB[:,:,1,2]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test2.png")
    plt.close()

    plot_sdata(fake_imagesBfromAt[:,:,1,2]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel+den from vel 2_$e ")
    plt.savefig("../plots/Shot_rec_df/vel+den test pred2_$e.png")
    plt.close()

    plot_sdata(XA[:,:,1,2]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel ")
    plt.savefig("../plots/Shot_rec_df/vel data test2.png")
    plt.close()

    plot_sdata(fake_imagesAfromBt[:,:,1,2]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel from vel+den 2_$e ")
    plt.savefig("../plots/Shot_rec_df/vel test pred2_$e.png")
    plt.close()



    plot_sdata(XB[:,:,1,3]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test3.png")
    plt.close()

    plot_sdata(fake_imagesBfromAt[:,:,1,3]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel+den from vel 3_$e ")
    plt.savefig("../plots/Shot_rec_df/vel+den test pred3_$e.png")
    plt.close()

    plot_sdata(XA[:,:,1,3]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel ")
    plt.savefig("../plots/Shot_rec_df/vel data test3.png")
    plt.close()

    plot_sdata(fake_imagesAfromBt[:,:,1,3]|>cpu,(0.8,1),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel from vel+den 3_$e ")
    plt.savefig("../plots/Shot_rec_df/vel test pred3_$e.png")
    plt.close()


    push!(mseofimb, Flux.mse(XB,fake_imagesBfromAt|>cpu))
    push!(mseofima, Flux.mse(XA,fake_imagesAfromBt|>cpu))

    if mod(e,10) == 0
      plt.plot(mseofima,label="domain A")
      plt.plot(mseofimb,label="domain B")
      plt.title("mseloss of testimg $e")
      plt.legend()
      plt.savefig("../plots/Shot_rec_df/mseofimg$e.png")
      plt.close()
    end
end


print("done training!!!")