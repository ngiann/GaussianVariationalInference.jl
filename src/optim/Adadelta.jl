mutable struct Adadelta

	ϵ::Float64
	ρ::Float64

	# Accumulate squares and of past gradients
	Eg²::Array{Float64,1}

	# Accumulate squares and of past updates
	EΔθ²::Array{Float64,1}

end


function Adadelta(ϵ, ρ, D)

	Adadelta(ϵ, ρ, zeros(D), zeros(D))

end


function step!(a::Adadelta, θ, g)
	
	# θ is the current parameter value
	# g is the gradient at θ

	RMS(x) = sqrt.(x .+ a.ϵ)

	ρ = a.ρ

	a.Eg²  = ρ*a.Eg²  + (1.0 - ρ) * (g .* g)

	Δθ 	   = - (RMS(a.EΔθ²)./ RMS(a.Eg²)) .* g

	a.EΔθ² = ρ*a.EΔθ² + (1.0 - ρ) * (Δθ .* Δθ)

	θ + Δθ

end

tostring(a::Adadelta) = @sprintf("Adadelta (ϵ = %g, ρ = %3f)", a.ϵ, a.ρ)