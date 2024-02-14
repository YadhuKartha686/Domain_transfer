
	ΔX, X, ΔY = generator.cond_net.backward_inv(((gs ./ factor)|>device), fake_images, invcall;)
    for i=1:generator.cond_net.L
        ΔY = generator.cond_net.squeezer.inverse(ΔY)
        Zy1 = generator.cond_net.squeezer.inverse(Zy1)
    end
    ΔY = generator.sum_net.backward(ΔY, Zy1) # unet has channels 1
    for i=1:generator.cond_net.L
        ΔY = generator.cond_net.squeezer.forward(ΔY)
        Zy1 = generator.cond_net.squeezer.forward(Zy1)
    end

