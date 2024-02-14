using DrWatson
@quickactivate "Domain_transfer"

using Flux
using LinearAlgebra
using Random
# using Metalhead
using JLD2
using Statistics
using ImageQualityIndexes 
using PyPlot
using SlimPlotting
using InvertibleNetworks
using MLUtils
# using BSON 
# using Wavelets
using StatsBase
using Distributions
using Images
using UNet

#### DATA LOADING #####
nx,ny = 256, 256
N = nx*ny;

data_path= "../data/CompassShotmid.jld2"
# datadir(CompassShot.jld2)
train_X = jldopen(data_path, "r")["X"]
train_y = jldopen(data_path, "r")["Y"]
  
train_xA = zeros(Float32, nx, ny, 1,900)
train_xB = zeros(Float32, nx, ny, 1,900)

  
for i=1:900
    sigma = 1.0
    train_xA[:,:,:,i] = imresize(imfilter(train_X[:,:,i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
    train_xB[:,:,:,i] = imresize(imfilter(train_X[:,:,900+i],KernelFactors.gaussian((sigma,sigma))),(nx,ny))
end



# Define the generator and discriminator networks

device = gpu #GPU does not accelerate at this small size. quicker on cpu
lr     = 5f-6
low = 0.5f0

# Architecture parametrs
chan_x = 1; chan_y = 1; L = 5; K = 10; n_hidden = 128 # Number of hidden channels in convolutional residual blocks

# Create network
G = NetworkConditionalGlow(chan_x, chan_y, n_hidden,  L, K; split_scales=true,activation=SigmoidLayer(low=low,high=1.0f0)) |> device;

model = Chain(
  # First convolutional layer
  Conv((7, 7), 1=>32, relu, pad=(3,3), stride=1),
  x -> maxpool(x, (2,2)), # Aggressive pooling to reduce dimensions
  
  # Second convolutional layer
  Conv((5, 5), 32=>64, relu, pad=(2,2), stride=1),
  x -> maxpool(x, (2,2)), # Further reduction
  
  # Third convolutional layer
  Conv((5, 5), 64=>64, relu, pad=1),
  x -> maxpool(x, (2,2)), # Final pooling to reduce to a very low dimension
  # Fourth convolutional layer
  Conv((3, 3), 64=>64, relu, pad=1),
  x -> maxpool(x, (2,2)), # Final pooling to reduce to a very low dimension

  Conv((3, 3), 64=>64, relu, pad=1),
  x -> maxpool(x, (2,2)), # Final pooling to reduce to a very low dimension
  
  # Flatten the output of the last convolutional layer before passing it to the dense layer
  Flux.flatten,
  
  # Fully connected layer
  Dense(3136, 512, relu),
  
  # Output layer for binary classification
  Dense(512, 1),
  sigmoid
)

# Summary Network
sum_net = true
h2      = nothing
unet_lev = 2
n_c = 1
n_in = 1
if sum_net
    h2 = Chain(Unet(n_c,n_in,unet_lev))
    trainmode!(h2, true)
    h2 = FluxBlock(h2)|> device
end


# Define the loss functions
function Dissloss(real_output, fake_output)
  real_loss = mean(Flux.binarycrossentropy.(real_output, 1f0))
  fake_loss = mean(Flux.binarycrossentropy.(fake_output, 0f0))
  return 0.5f0*(real_loss + fake_loss)
end

function Genloss(fake_output) 
  return mean(Flux.binarycrossentropy.(fake_output, 1f0))
end


# Initialize networks and optimizers
generator = SummarizedNet(G, h2) |> gpu
discriminatorA = gpu(model)
discriminatorB = gpu(model)

# generator = G
# discriminatorA = model
# discriminatorB = model

clipnorm_val = 5f0
optimizer_g = Flux.Optimiser(ClipNorm(clipnorm_val), ADAM(lr))
lrd = 1f-6
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
factor = 1f0

function z_shape_simple(G, ZX_test)
  Z_save, ZX = split_states(ZX_test[:], G.Z_dims)
  for i=G.L:-1:1
      if i < G.L
          ZX = tensor_cat(ZX, Z_save[i])
      end
      ZX = G.squeezer.inverse(ZX) 
  end
  ZX
end

function z_shape_simple_forward(G, X)
  G.split_scales && (Z_save = array_of_array(X, G.L-1))
  for i=1:G.L
      (G.split_scales) && (X = G.squeezer.forward(X))
      if G.split_scales && i < G.L    # don't split after last iteration
          X, Z = tensor_split(X)
          Z_save[i] = Z
          G.Z_dims[i] = collect(size(Z))
      end
  end
  G.split_scales && (X = cat_states(Z_save, X))
  return X
end


n_epochs     = 250
for e=1:n_epochs# epoch loop
  epoch_loss_diss=0.0
  epoch_loss_gen=0.0
  idx_eA = reshape(randperm(n_train), imgs, n_batches)
  idx_eB = reshape(randperm(n_train), imgs, n_batches)
  for b = 1:n_batches # batch loop
        @time begin
          ############# Loading domain A data ############## 
          
          XA = train_xA[:, :, :, idx_eA[:,b]] + randn(Float32,(nx,ny,1,imgs)) ./1f5
          XB = train_xB[:, :, :, idx_eA[:,b]] + randn(Float32,(nx,ny,1,imgs)) ./1f5
          X = cat(XA, XB,dims=4)
          Y = cat(YA, YB,dims=4)

          Zx, Zy, lgdet = generator.forward(X|> device, Y|> device)  #### concat so that network normalizes ####

          ######## interchanging conditions to get domain transferred images during inverse call #########
          # Zx = z_shape_simple(G,Zx)
          # Zy = z_shape_simple(G,Zy)

          ZyA = Zy[:,:,:,1:imgs]
          ZyB = Zy[:,:,:,imgs+1:end]

          ZxA = Zx[:,:,:,1:imgs]
          ZxB = Zx[:,:,:,imgs+1:end]

          Zy1 = cat(ZyB,ZyA,dims=4)

          # Zx = z_shape_simple_forward(generator,Zx)
          # Zy = z_shape_simple_forward(generator,Zy)
          # Zy1 = z_shape_simple_forward(generator,Zy1)
          
          # Zx = reshape(Zx,(nx,ny,1,imgs*2))
          # Zy = reshape(Zy,(8,8,1024,imgs*2))
          # Zy1 = reshape(Zy1,(8,8,1024,imgs*2))

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
                    
          gsA = gradient(x -> Genloss(discriminatorA(x|> device)), fake_imagesAfromB)[1]  #### getting gradients wrt A fake ####
          gsB = gradient(x -> Genloss(discriminatorB(x|> device)), fake_imagesBfromA)[1]  #### getting gradients wrt B fake ####
          

          gs = cat(gsB,gsA,dims=4)
     
          generator.backward_inv(((gs ./ factor)|>device), fake_images, invcall;Y_save=Zy1|>device) #### updating grads wrt image ####
          generator.backward(Zx / imgs*2, Zx, Zy;Y_save=Y|>device)
   
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
          lossA = Genloss(fake_outputA)  #### log(1 - D(fake)) + mse ####
          lossB = Genloss(fake_outputB)  #### log(1 - D(fake)) + mse ####
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
              "; dissloss = ", (lossAd+lossBd) , 
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

    # Zx = z_shape_simple(generator,Zx)
    # Zy = z_shape_simple(generator,Zy)

              ######## interchanging conditions to get domain transferred images during inverse call #########

    ZyA = Zy[:,:,:,1:imgs]
    ZyB = Zy[:,:,:,imgs+1:end]

    Zy = cat(ZyB,ZyA,dims=4)

    # Zx = z_shape_simple_forward(generator,Zx)
    # Zy = z_shape_simple_forward(generator,Zy)

          
    # Zx = reshape(Zx,(nx,ny,1,imgs*2))
    # Zy = reshape(Zy,(8,8,1024,imgs*2))


    fake_images,invcall = generator.inverse(Zx|>device,Zy)  ###### generating images #######

              ####### getting fake images from respective domain ########

    fake_imagesAfromBt = fake_images[:,:,:,imgs+1:end]
    fake_imagesBfromAt = fake_images[:,:,:,1:imgs]

    plot_sdata(XB[:,:,1,1]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test1.png")
    plt.close()

    plot_sdata(fake_imagesBfromAt[:,:,1,1]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel+den from vel 1_$e ")
    plt.savefig("../plots/Shot_rec_df/vel+den test pred1_$e.png")
    plt.close()

    plot_sdata(XA[:,:,1,1]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel ")
    plt.savefig("../plots/Shot_rec_df/vel data test1.png")
    plt.close()

    plot_sdata(fake_imagesAfromBt[:,:,1,1]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel from vel+den 1_$e ")
    plt.savefig("../plots/Shot_rec_df/vel test pred1_$e.png")
    plt.close()



    plot_sdata(XB[:,:,1,2]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test2.png")
    plt.close()

    plot_sdata(fake_imagesBfromAt[:,:,1,2]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel+den from vel 2_$e ")
    plt.savefig("../plots/Shot_rec_df/vel+den test pred2_$e.png")
    plt.close()

    plot_sdata(XA[:,:,1,2]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel ")
    plt.savefig("../plots/Shot_rec_df/vel data test2.png")
    plt.close()

    plot_sdata(fake_imagesAfromBt[:,:,1,2]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel from vel+den 2_$e ")
    plt.savefig("../plots/Shot_rec_df/vel test pred2_$e.png")
    plt.close()



    plot_sdata(XB[:,:,1,3]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel+den ")
    plt.savefig("../plots/Shot_rec_df/vel+den data test3.png")
    plt.close()

    plot_sdata(fake_imagesBfromAt[:,:,1,3]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title.(" pred vel+den from vel 3_$e ")
    plt.savefig("../plots/Shot_rec_df/vel+den test pred3_$e.png")
    plt.close()

    plot_sdata(XA[:,:,1,3]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
    plt.title("data test vel ")
    plt.savefig("../plots/Shot_rec_df/vel data test3.png")
    plt.close()

    plot_sdata(fake_imagesAfromBt[:,:,1,3]|>cpu,(7.03,2.488),perc=95,vmax=0.03,cbar=true)
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

    if mod(e,10) == 0
      Params = get_params(generator) |> cpu;
      save_dict = @strdict e L K n_hidden lr 
      @tagsave(
           "../plots/Shot_rec_df/"*savename(save_dict, "jld2"; digits=6),
           save_dict;
           safe=true
      )
    end
end


print("done training!!!")


# function get_network(path)
#   # test parameters
#   batch_size = 32
#   n_post_samples = 128
#   device = gpu
#   #load previous hyperparameters
#   data_path = datadir("L_inifinity_norm_cruyff_training_30_July/" * path);
#   bson_file = BSON.load(data_path);
#   n_hidden = bson_file["n_hidden"];
#   L = bson_file["L"];
#   K = bson_file["K"];
#   Params = bson_file["Params"];
#   e = bson_file["e"];
#   noise_lev_x = bson_file["noise_lev_x"];
#   n_train = bson_file["n_train"];
#   G = NetworkConditionalGlow(1, 1, n_hidden,  L, K; split_scales=true, activation=SigmoidLayer(low=0.5f0,high=1.0f0));
#   p_curr = get_params(G);
#   for p in 1:length(p_curr)
#   p_curr[p].data = Params[p].data
#   end
#   return G
# end