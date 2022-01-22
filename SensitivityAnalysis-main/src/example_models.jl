#=
Systems of ODEs

This section includes the systems of ODEs used as examples in the visualizations
part of this code.
=#

"""
    ROBER(dx, x, p, t)
System of ODEs for the ROBER model.  This is a stiff system of equations.
"""
function ROBER(dx, x, p, t)
  dx[1] = -p[1]*x[1]+p[3]*x[2]*x[3]
  dx[2] =  p[1]*x[1]-p[2]*x[2]^2-p[3]*x[2]*x[3]
  dx[3] =  p[2]*x[2]^2
end

"""
    CARRGO(dx, x, p, t)

System of ODEs for the CARRGO model.  Compatible with the methods in this code
as well as DifferentialEquations.jl
"""
function CARRGO(dx, x, p, t)
    #x1 = cancer cells; x2 = immune cells
    dx[1] = p[4] * x[1] * (1 - x[1] / p[5]) - p[1] * x[1] * x[2]
    dx[2] = p[2]* x[1] * x[2] - p[3] * x[2]
end

"""
    SIR(dx, x, p, t)

System of ODEs for the SIR model.  Compatible with the methods in this code
as well as DifferentialEquations.jl
"""
function SIR(dx, x, p, t)
    #x1 = susceptible; x2 = infected; x3 = recovered
    dx[1] = -p[1] * x[1] * x[2] / p[3]
    dx[2] = p[1] * x[1] * x[2] / p[3] - p[2] * x[2]
    dx[3] = p[2] * x[2]
end

"""
    LotkaVolterra(dx, x, p, t)

System of ODEs for the Lotka-Volterra model.  Compatible with the methods in this code
as well as DifferentialEquations.jl
"""
function LotkaVolterra(dx, x, p, t)
    #x1 = prey; x2 = predator
  dx[1] = p[1] * x[1] - p[2] * x[1] * x[2]
  dx[2] = -p[4] * x[2] + p[3] * x[1] * x[2]
end

"""
    Vaccine(dx, x, p, t)

System of ODEs for the vaccine model from Alfonso Landeros.  Compatible with
the methods in this code as well as DifferentialEquations.jl
"""
function Vaccine(dx, x, p, t)
  #x1 = S_0; x2 = S_1; x3 = S_2; x4 = I_0; x5 = I_1; x6 = I_2; x7 = R
  #p1 = lambda_0; p2 = nu_0, p3 = delta_1, p4 = delta_2, p5 = rho, p6 = lambda_1; p7 = total vaccine; p8 = lambda_2; p9 = gamma
  dx[1] = -(p[1]*(x[4]+x[5]+x[6])+p[2])*x[1] + p[3]*x[2] + p[4]*x[3]+p[5]*x[7]
  dx[2] = -(p[6]*(x[4]+x[5]+x[6])+(p[7]-p[6])+p[3])*x[2]+p[2]*x[1]
  dx[3] = -(p[8]*(x[4]+x[5]+x[6]) + p[4])*x[3] + (p[7]-p[6])*x[2]
  dx[4] = -p[9]*x[4] + p[1]*x[1]*(x[4]+x[5]+x[6])
  dx[5] = -p[9]*x[5] + p[6]*x[2]
  dx[6] = -p[9]*x[6]+p[8]*x[3]
  dx[7] = -p[5]*x[7]+p[9]*(x[4]+x[5]+x[6])
end

"""
    MCC(dx, x, p, t)

System of ODEs for Mammalian Cell Cycle model from https://www.ebi.ac.uk/biomodels/BIOMD0000000730 .
"""
function MCC(dx, x, p, t)
    (pRBc1, pRBc2, Cd, Mdi, Md, pRB, E2F, pRBp, AP1, p27, Mdp27) = (x[1], x[2], x[3], x[4], x[5], x[6], 
                                                                    x[7], x[8], x[9], x[10], x[11])
    (kpc1, kpc3, kcd1, kdecom1, kcom1, CDK4_tot, Vm2d, k2d, Vm1d, k1d, kcd2, ki7, ki8, kcd1, kc1) = (p[1], p[2], p[3], 
                                                                                                     p[4], p[5], p[6], 
                                                                                                     p[7], p[8], p[9], 
                                                                                                     p[10], p[11], p[12], 
                                                                                                     p[13], p[14], p[15])
    dx[1] = kpc1 * pRB * E2F
    dx[2] = kpc3 * pRBp * E2F
    dx[3] = kcd1 * AP1 + kdecom1 * Mdi - kcom1 * Cd * (CDK4_tot - (Mdi + Md + Mdp27))
    dx[4] = Vm2d * Md / (k2d + Md) + 2 * kcom1 * Cd * (CDK4_tot - (Mdi + Md + Mdp27))
    dx[5] = Vm1d * Mdi / (k1d + Mdi) + kcom1 + Cd * (CDK4_tot - (Mdi + Md + Mdp27))
    dx[6] = kcd2 * E2F * ki7 / (ki7 + pRB) * ki8 / (ki8 + pRBp)
    dx[7] = kcd2 * E2F * ki7 / (ki7 + pRB) * ki8 / (ki8 + pRBp)
    dx[8] = kcd2 * E2F * ki7 / (ki7 + pRB) * ki8 / (ki8 + pRBp)
    dx[9] = kcd1 * AP1
    dx[10] = 0.0
    dx[11] = kc1 * Md * p27 + kcom1 * Cd * (CDK4_tot - (Mdi + Md + Mdp27))
end
