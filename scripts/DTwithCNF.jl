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

data_path= "../data/CompassShot.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
train_xA = zeros(Float32, 2048, 512, 1,1)
train_xB = zeros(Float32, 2048, 512, 1,1)

train_YA = ones(Float32,2048,512,1,1)
train_YB = ones(Float32,2048,512,1,1).*2


indices_of_A = findall(x -> x == 0.0, train_y[:,1])
indices_of_B = findall(x -> x == 1.0, train_y[:,1])

# for i=1:148
#     sigma = 1.0
#     test_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_A[752+i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
#     test_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_B[752+i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
# end 

  
for i=1:1
    sigma = 1.0
    train_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_A[i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
    train_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,indices_of_B[i]],KernelFactors.gaussian((sigma,sigma))),(2048,512))
end


batch_size = 1
nx,ny = 2048, 512
N = nx*ny;

n_train = 752

n_batches = cld(n_train,batch_size)


# Define the generator and discriminator networks

n_epochs     = 10000
device = gpu
lr     = 4f-3
lr_step   = 10
lr_rate = 0.75f0
clipnorm_val = 10f0
noise_lev_x  = 0.005f0
noise_lev_y  = randn(Float32,(2048,512,1,batch_size))./1000
split_scales = true

# Architecture parametrs
chan_x = 1; chan_y = 1; L = 2; K = 10; n_hidden = 32 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K; split_scales=true) |> device;


model = Chain(
    Chain([
      Conv((7, 7), 1 => 64, pad=3, stride=2, bias=false),  # 9_408 parameters
      BatchNorm(64, relu),            # 128 parameters, plus 128
      MaxPool((3, 3), pad=1, stride=2),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),        # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),              # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64, relu),        # 128 parameters, plus 128
          Conv((3, 3), 64 => 64, pad=1, bias=false),  # 36_864 parameters
          BatchNorm(64),              # 128 parameters, plus 128
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 64 => 128, pad=1, stride=2, bias=false),  # 73_728 parameters
          BatchNorm(128, relu),       # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ),
        Chain([
          Conv((1, 1), 64 => 128, stride=2, bias=false),  # 8_192 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128, relu),       # 256 parameters, plus 256
          Conv((3, 3), 128 => 128, pad=1, bias=false),  # 147_456 parameters
          BatchNorm(128),             # 256 parameters, plus 256
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 128 => 256, pad=1, stride=2, bias=false),  # 294_912 parameters
          BatchNorm(256, relu),       # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ),
        Chain([
          Conv((1, 1), 128 => 256, stride=2, bias=false),  # 32_768 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256, relu),       # 512 parameters, plus 512
          Conv((3, 3), 256 => 256, pad=1, bias=false),  # 589_824 parameters
          BatchNorm(256),             # 512 parameters, plus 512
        ),
        identity,
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 256 => 512, pad=1, stride=2, bias=false),  # 1_179_648 parameters
          BatchNorm(512, relu),       # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ),
        Chain([
          Conv((1, 1), 256 => 512, stride=2, bias=false),  # 131_072 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ]),
      ),
      Parallel(
        Metalhead.addrelu,
        Chain(
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512, relu),       # 1_024 parameters, plus 1_024
          Conv((3, 3), 512 => 512, pad=1, bias=false),  # 2_359_296 parameters
          BatchNorm(512),             # 1_024 parameters, plus 1_024
        ),
        identity,
      ),
    ]),
    Chain(
      AdaptiveMeanPool((1, 1)),
      MLUtils.flatten,
      Dense(512 => 1, sigmoid),             # 513_000 parameters
    ),
  )



# Define the loss functions
function Dissloss(real_output, fake_output)
  real_loss = mean(Flux.binarycrossentropy.(real_output, 1f0))
  fake_loss = mean(Flux.binarycrossentropy.(fake_output, 0f0))
  return (real_loss + fake_loss)
end

Genloss(fake_output) = mean(Flux.binarycrossentropy.(fake_output, 1f0))


# Initialize networks and optimizers
generator = G |> gpu
discriminatorA = gpu(model)
discriminatorB = gpu(model)

# generator = G
# discriminatorA = model
# discriminatorB = model

opt_adam = "adam"
optimizer_g = Flux.ADAM(lr)
optimizer_da = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))
optimizer_db = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))
genloss=[]
dissloss = []

XA = train_xA[:,:,:,1:1]
XB = train_xB[:,:,:,1:1]
Z_fix =  randn(Float32,2048,512,1,2)
YA = randn(Float32,size(XA))
YB = randn(Float32,size(XB))

lossnrm      = []; logdet_train = []; 
factor = 1f-20


for e=1:n_epochs# epoch loop
  epoch_loss_diss=0.0
  epoch_loss_gen=0.0
    @time begin

          ############# Loading domain A data ##############    

          X = cat(XB, XA,dims=4)
          Y = cat(YA, YB,dims=4)
          Zx, Zy, lgdet = generator.forward(Z_fix|> device, Y|> device)  #### concat so that network normalizes ####

          ######## interchanging conditions to get domain transferred images during inverse call #########

          ZyA = Zy[:,:,:,1:1]
          ZyB = Zy[:,:,:,2:2]

          Zy = cat(ZyB,ZyA,dims=4)

          fake_images,invcall = generator.inverse(Z_fix|>device,Zy)  ###### generating images #######

          ####### getting fake images from respective domain ########

          fake_imagesAfromB = fake_images[:,:,:,2:2]
          fake_imagesBfromA = fake_images[:,:,:,1:1]

          invcallA = invcall[:,:,:,2:2]
          invcallB = invcall[:,:,:,1:1]

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
                    
          gsA = gradient(x -> Genloss(discriminatorA(x|> device)), fake_imagesAfromB)[1]  #### getting gradients wrt A fake ####
          gsB = gradient(x -> Genloss(discriminatorB(x|> device)), fake_imagesBfromA)[1]  #### getting gradients wrt B fake ####

          generator.backward_inv(((gsA ./ factor)|>device), fake_imagesAfromB, invcallA;) #### updating grads wrt A ####
          generator.backward_inv(((gsB ./ factor)|>device), fake_imagesBfromA, invcallB;) #### updating grads wrt B ####

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
          lossA = Genloss(fake_outputA)  #### log(1 - D(fake)) ####
          lossB = Genloss(fake_outputB)  #### log(1 - D(fake)) ####
          f_all = 0
          for i in 1:2
            X_gen_cpu = fake_images|>cpu
            g = (X_gen_cpu[:,:,1,i] .- X)  
            f = norm(g)^2
            # gs[:,:,:,i] =  g
            f_all += f
          end
          loss = lossA + lossB #+ ml

          append!(lossnrm, f_all / 2)  # normalize by image size and batch size
          append!(logdet_train, (-lgdet) / N) # logdet is internally normalized by batch size
          append!(genloss, loss)  # normalize by image size and batch size
          append!(dissloss, (lossAd+lossBd)/2 ) # logdet is internally normalized by batch size


          epoch_loss_diss += (lossAd+lossBd)/2
          epoch_loss_gen += loss

          println("Iter: epoch=", e, "/", n_epochs,
            "; genloss = ",  loss, 
              "; dissloss = ", (lossAd+lossBd)/2 , 
              "; f l2 = ",  lossnrm[end], 
              "; lgdet = ", logdet_train[end], "\n")

          Base.flush(Base.stdout)

    end
      if mod(e,100) == 0
            plot_sdata(XA[:,:,:,1],(0.8,1),vmax=0.04f0,perc=95,cbar=true)
            plt.title("Shot record train ( vel + den) $e")
            plt.savefig("../plots/Shot_rec_df/vel+den train$e.png")
            plt.close()
    
            plot_sdata(fake_imagesAfromB[:,:,:,1]|>cpu,(0.8,1),vmax=0.04f0,perc=95,cbar=true)
            plt.title("Shot record pred A from B $e")
            plt.savefig("../plots/Shot_rec_df/vel$e.png")
            plt.close()

            plot_sdata(fake_imagesBfromA[:,:,:,1]|>cpu,(0.8,1),vmax=0.04f0,perc=95,cbar=true)
            plt.title("Shot record pred B from A $e")
            plt.savefig("../plots/Shot_rec_df/vel+den$e.png")
            plt.close()

            plt.plot(lossnrm)
            plt.title("loss $e")
            plt.savefig("../plots/Shot_rec_df/lossnorm$e.png")
            plt.close()
            plt.plot(logdet_train)
            plt.title("logdet $e")
            plt.savefig("../plots/Shot_rec_df/logdet$e.png")
            plt.close()
            plt.plot(1:e,genloss[1:e])
            plt.title("genloss $e")
            plt.savefig("../plots/Shot_rec_df/genloss$e.png")
            plt.close()
            plt.plot(1:e,dissloss[1:e])
            plt.title("dissloss $e")
            plt.savefig("../plots/Shot_rec_df/dissloss$e.png")
            plt.close()
      end
end

# Main training loop
# for e=1:n_epochs# epoch loop
#     epoch_loss_diss=0.0
#     epoch_loss_gen=0.0
#     idx_eA = reshape(randperm(n_train), batch_size, n_batches)
#     idx_eB = reshape(randperm(n_train), batch_size, n_batches)
#     for b = 1:n_batches # batch loop
#     	@time begin

#             ############# Loading domain A data ###############

#             XA = train_xA[:, :, :, idx_eA[:,b]];
#             YA = train_YA[:, :, :, idx_eA[:,b]];
#             XA .+= noise_lev_x*randn(Float32, size(XA));
#             YA = YA + noise_lev_y;

#             ############# Loading domain B data ###############

#             XB = train_xB[:, :, :, idx_eB[:,b]];
#             YB = train_YB[:, :, :, idx_eB[:,b]];
#             XB .+= noise_lev_x*randn(Float32, size(XB));
#             YB = YB + noise_lev_y;
      

#             X = cat(XA, XB,dims=4)
#             Y = cat(YA, YB,dims=4)
#             Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####

#             ######## interchanging conditions to get domain transferred images during inverse call #########

#             ZyA = Zy[:,:,:,1:4]
#             ZyB = Zy[:,:,:,5:end]

#             Zy = cat(ZyB,ZyA,dims=4)

#             fake_images,invcall = generator.inverse(Ztest|>device,Zy)  ###### generating images #######

#             ####### getting fake images from respective domain ########

#             fake_imagesAfromB = fake_images[:,:,:,5:end]
#             fake_imagesBfromA = fake_images[:,:,:,1:4]

#             invcallA = invcall[:,:,:,5:end]
#             invcallB = invcall[:,:,:,1:4]

#             ####### discrim training ########

#             dA_grads = Flux.gradient(Flux.params(discriminatorA)) do
#                 real_outputA = discriminatorA(XA|> device)
#                 fake_outputA = discriminatorA(fake_imagesAfromB|> device)
#                 lossA = Dissloss(real_outputA, fake_outputA)
#             end
#             Flux.Optimise.update!(optimizer_da, Flux.params(discriminatorA),dA_grads)  #### domain A discrim ####

            
#             dB_grads = Flux.gradient(Flux.params(discriminatorB)) do
#                 real_outputB = discriminatorB(XB|> device)
#                 fake_outputB = discriminatorB(fake_imagesBfromA|> device)
#                 lossB = Dissloss(real_outputB, fake_outputB)
#             end
#             Flux.Optimise.update!(optimizer_db, Flux.params(discriminatorB),dB_grads)  #### domain B discrim ####

#             ## minlog (1-D(fakeimg)) <--> max log(D(fake)) + norm(Z)
                      
#             gsA = gradient(x -> Genloss(discriminatorA(x|> device)), fake_imagesAfromB)[1]  #### getting gradients wrt A fake ####
#             gsB = gradient(x -> Genloss(discriminatorB(x|> device)), fake_imagesBfromA)[1]  #### getting gradients wrt B fake ####

#             generator.backward_inv(((gsA ./ factor)|>device), fake_imagesAfromB, invcallA;) #### updating grads wrt A ####
#             generator.backward_inv(((gsB ./ factor)|>device), fake_imagesBfromA, invcallB;) #### updating grads wrt B ####

#             for p in get_params(generator)
#                 Flux.update!(optimizer_g,p.data,p.grad)
#             end
#             clear_grad!(generator) #### updating generator ####

#             #loss calculation for printing
#             fake_outputA = discriminatorA(fake_imagesAfromB|> device)
#             fake_outputB = discriminatorB(fake_imagesBfromA|> device)
#             real_outputA = discriminatorA(XA|> device)
#             real_outputB = discriminatorB(XB|> device)

#             lossAd = Dissloss(real_outputA, fake_outputA)  #### log(D(real)) + log(1 - D(fake)) ####
#             lossBd = Dissloss(real_outputB, fake_outputB)  #### log(D(real)) + log(1 - D(fake)) ####
#             lossA = Genloss(fake_outputA)  #### log(1 - D(fake)) ####
#             lossB = Genloss(fake_outputB)  #### log(1 - D(fake)) ####
#             ml = norm(Zx)^2/(N*batch_size)
#             loss = lossA + lossB #+ ml

#             append!(lossnrm, ml)  # normalize by image size and batch size
# 	          append!(logdet_train, (-lgdet) / N) # logdet is internally normalized by batch size


#             epoch_loss_diss += (lossAd+lossBd)/2
#             epoch_loss_gen += loss

#             println("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
# 	            "; genloss = ",  loss, 
#                 "; dissloss = ", (lossAd+lossBd)/2 , 
#                 "; f l2 = ",  lossnrm[end], 
#                 "; lgdet = ", logdet_train[end], "\n")

#             Base.flush(Base.stdout)

#             plt.plot(lossnrm)
#             plt.title("loss $b")
#             plt.savefig("../plots/Shot_rec_df/lossnorm$e.png")
#             plt.close()
#             plt.plot(logdet_train)
#             plt.title("logdet $b")
#             plt.savefig("../plots/Shot_rec_df/logdet$e.png")
#             plt.close()

#         end    
#     end
#     avg_epoch_lossd = epoch_loss_diss / size(idx_eA, 2)
#     avg_epoch_lossg= epoch_loss_gen / size(idx_eA, 2)
#     push!(genloss, avg_epoch_lossg)
#     push!(dissloss, avg_epoch_lossd)
#     plt.plot(1:e,genloss[1:e])
#     plt.title("genloss $e")
#     plt.savefig("../plots/Shot_rec_df/genloss$e.png")
#     plt.close()
#     plt.plot(1:e,dissloss[1:e])
#     plt.title("dissloss $e")
#     plt.savefig("../plots/Shot_rec_df/dissloss$e.png")
#     plt.close()
    
#     if n_epochs % 1 == 0
        
#         # Optionally, generate and save a sample image during training
#         XA = train_xA[:, :, :, 1:4];
#         YA = train_YA[:, :, :, 1:4];
#         XA .+= noise_lev_x*randn(Float32, size(XA));
#         YA = YA + noise_lev_y;

#             ############# Loading domain B data ###############

#         XB = train_xB[:, :, :, 1:4];
#         YB = train_YB[:, :, :, 1:4];
#         XB .+= noise_lev_x*randn(Float32, size(XB));
#         YB = YB + noise_lev_y;
      

#         X = cat(XA, XB,dims=4)
#         Y = cat(YA, YB,dims=4)
#         _, Zy, _ = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####


#         ZyA = Zy[:,:,:,1:4]
#         ZyB = Zy[:,:,:,5:end]

#         Zy = cat(ZyB,ZyA,dims=4)

#         fake_images,invcall = generator.inverse(Ztest|>device,Zy)  ###### generating images #######

#             ####### getting fake images from respective domain ########

#         fake_imagesAfromB = fake_images[:,:,:,5:end]
#         fake_imagesBfromA = fake_images[:,:,:,1:4]

#         plot_sdata(train_xB[:,:,:,1]|> cpu,(0.8,1),vmax=0.04f0,perc=95,cbar=true)
#         plt.title("Shot record (vel+den) $e")
#         plt.savefig("../plots/Shot_rec_df/vel+dentrain$e.png")
#         plt.close()

#         plot_sdata(train_xA[:,:,:,1]|> cpu,(0.8,1),vmax=0.04f0,perc=95,cbar=true)
#         plt.title("Shot record pred vel) $e")
#         plt.savefig("../plots/Shot_rec_df/veltrain$e.png")
#         plt.close()
#         # Save or visualize the generated_image as needed

#         plot_sdata(fake_imagesAfromB[:,:,:,1]|> cpu,(0.8,1),vmax=0.04f0,perc=95,cbar=true)
#         plt.title("Shot record pred ( vel from vel+den) $e")
#         plt.savefig("../plots/Shot_rec_df/vel+den$e.png")
#         plt.close()

#         plot_sdata(fake_imagesBfromA[:,:,:,1]|> cpu,(0.8,1),vmax=0.04f0,perc=95,cbar=true)
#         plt.title("Shot record pred ( vel+den from vel) $e")
#         plt.savefig("../plots/Shot_rec_df/vel$e.png")
#         plt.close()
#         # Save or visualize the generated_image as needed
#     end
# end
