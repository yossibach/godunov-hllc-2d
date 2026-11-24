#################################
import numpy as np
from numpy import sqrt, sin, cos, pi, exp, log
import matplotlib.pyplot as plt
import sys

def hllc(rho_L,rho_R,p_L,p_R,u_L,u_R,v_L,v_R,gamma):
    # Toro chapter 10.6.1
    # step 1
    a_L = sqrt(gamma*p_L/rho_L)
    a_R = sqrt(gamma*p_R/rho_R)
    S_L = min(u_L-a_L, u_R-a_R)    # Toro 10.38
    S_R = max(u_L+a_L, u_R+a_R)
    Sstar = (p_R-p_L+rho_L*u_L*(S_L-u_L)-rho_R*u_R*(S_R-u_R))/(rho_L*(S_L-u_L)-rho_R*(S_R-u_R))  # Toro 10.58
    pstar = p_L + rho_L*(S_L-u_L)*(Sstar-u_L)   # Toro 10.57, assuming p*=p*L=p*R
    
    # step 2
    E_L   = p_L/(gamma-1) + 0.5*rho_L*(u_L**2+v_L**2)
    U1star_L = rho_L*((S_L-u_L)/(S_L-Sstar))  # Toro 10.33
    U2star_L = U1star_L * Sstar
    U3star_L = U1star_L * v_L
    U4star_L = U1star_L * (E_L/rho_L + (Sstar-u_L)*(Sstar + p_L/rho_L/(S_L-u_L)))
    
    E_R   = p_R/(gamma-1) + 0.5*rho_R*(u_R**2+v_R**2)
    U1star_R = rho_R*((S_R-u_R)/(S_R-Sstar))
    U2star_R = U1star_R * Sstar
    U3star_R = U1star_R * v_R
    U4star_R = U1star_R * (E_R/rho_R + (Sstar-u_R)*(Sstar + p_R/rho_R/(S_R-u_R)))
    
    U1_L = rho_L
    U2_L = rho_L*u_L
    U3_L = rho_L*v_L
    U4_L = E_L
    
    U1_R = rho_R
    U2_R = rho_R*u_R
    U3_R = rho_R*v_R
    U4_R = E_R
    
    # step 3
    F1_L = rho_L*u_L          # Toro 10.5
    F2_L = rho_L*u_L**2 + p_L
    F3_L = rho_L*u_L*v_L
    F4_L = u_L*( E_L + p_L )

    F1_R = rho_R*u_R
    F2_R = rho_R*u_R**2 + p_R
    F3_R = rho_R*u_R*v_R
    F4_R = u_R*( E_R + p_R )
        
    F1star_L = F1_L + S_L*( U1star_L - U1_L )  # Toro 10.26
    F2star_L = F2_L + S_L*( U2star_L - U2_L )
    F3star_L = F3_L + S_L*( U3star_L - U3_L )
    F4star_L = F4_L + S_L*( U4star_L - U4_L )
    
    F1star_R = F1_R + S_R*( U1star_R - U1_R ) # Toro 10.28
    F2star_R = F2_R + S_R*( U2star_R - U2_R )
    F3star_R = F3_R + S_R*( U3star_R - U3_R )
    F4star_R = F4_R + S_R*( U4star_R - U4_R )
    
    # print('S_L= ',S_L,', Sstar = ',Sstar,' , S_R=',S_R)
    if 0<=S_L:                # Toro 10.34
        F1_hllc = F1_L
        F2_hllc = F2_L
        F3_hllc = F3_L
        F4_hllc = F4_L
    
    if (S_L<=0 and 0<=Sstar):
        F1_hllc = F1_L + S_L*( U1star_L - rho_L     )   # U1=rho
        F2_hllc = F2_L + S_L*( U2star_L - rho_L*u_L )  # U2 = rho*u
        F3_hllc = F3_L + S_L*( U3star_L - rho_L*v_L )  # U3 = rho*v
        F4_hllc = F4_L + S_L*( U4star_L - E_L       )  # U4 = E
        
    if (Sstar<=0 and 0<=S_R):
        F1_hllc = F1_R + S_R*( U1star_R - rho_R     )   # U1=rho
        F2_hllc = F2_R + S_R*( U2star_R - rho_R*u_R )  # U2 = rho*u
        F3_hllc = F3_R + S_R*( U3star_R - rho_R*v_R )  # U2 = rho*u
        F4_hllc = F4_R + S_R*( U4star_R - E_R       )  # U3 = E
        
    if 0>=S_R:
        F1_hllc = F1_R
        F2_hllc = F2_R
        F3_hllc = F3_R
        F4_hllc = F4_R 
        # print('deb d')
        
    Fhllc = np.array([F1_hllc,F2_hllc,F3_hllc,F4_hllc])
    return Fhllc

###############################################################

def IC_kh_springel(L,W,N,M):  # KH Springel 2010
    rho_i,u_i  = 2, 0.5 # 10, 0.5 # 
    rho_o,u_o  = 1,-0.5
    p          = 2.5
    w0,twosig2 = 0.1, 0.05**2
    
    dx    = L/N
    dy    = W/M
    x     = np.arange(M)*dx
    y     = np.arange(N)*dy
    
    rho   = np.ones((M,N))
    p     = np.ones((M,N))
    u     = np.ones((M,N))
    v     = np.ones((M,N))

    X = np.arange(0, L, dx)
    Y = np.arange(0, W, dy)
    X, Y = np.meshgrid(X, Y)
    inner = abs(X-0.5) < 0.25            # the dense central band
    rho   = np.where(inner, rho_i, rho_o)
    u     = np.where(inner, u_i,   u_o)
    v   = w0*sin(4*pi*Y)*( exp(-(X-0.25)**2/twosig2) + exp(-(X-0.75)**2/twosig2) )
    p   = 2.5*np.ones((M,N))
    
    return(x,y,dx,dy,rho,p,u,v)

def IC_sod_explosion(L, W, N, M):
    """Circular 2D Sod-like explosion: a high-pressure disk in the centre (use gamma=1.4)."""
    rho_in,  p_in  = 1.0,   1.0
    rho_out, p_out = 0.125, 0.1
    r0 = 0.2*L                               # disk radius

    dx = L/N
    dy = W/M
    x  = np.arange(M)*dx
    y  = np.arange(N)*dy

    X, Y   = np.meshgrid(np.arange(N)*dx, np.arange(M)*dy)
    inside = np.hypot(X - L/2, Y - W/2) < r0
    rho = np.where(inside, rho_in, rho_out)
    p   = np.where(inside, p_in,   p_out)
    u   = np.zeros((M,N))
    v   = np.zeros((M,N))
    return(x,y,dx,dy,rho,p,u,v)

def IC_kh_sharp(L, W, N, M):
    """KH with a sharp, sinusoidally displaced interface between two streams."""
    rho_1, u_1 = 1.0,  1.0                   # beyond the interface
    rho_2, u_2 = 0.1, -1.0                   # before the interface
    p0   = 1.0
    A, k = 0.04*L, 1                         # displacement amplitude, number of wavelengths

    dx = L/N
    dy = W/M
    x  = np.arange(M)*dx
    y  = np.arange(N)*dy

    X, Y  = np.meshgrid(np.arange(N)*dx, np.arange(M)*dy)
    upper = (X - L/2) > A*np.sin(2*np.pi*k*Y/W)
    rho = np.where(upper, rho_1, rho_2)
    u   = np.where(upper, u_1,   u_2)
    v   = np.zeros((M,N))
    p   = p0*np.ones((M,N))
    return(x,y,dx,dy,rho,p,u,v)

def BCrigid(rho,p,u,v):   # vertical rigid boundary conditions
    rho[ :,0]  = rho[:, 1]
    rho[:,-1]  = rho[:,-2]
    p[:, 0]    =  p[:, 1]
    p[:,-1]    =  p[:,-2]
    u[:, 0]    =  u[:, 1]
    u[:,-1]    =  u[:,-2]
    v[:, 0]    = -v[:, 1]
    v[:,-1]    = -v[:,-2]

    return(rho,p,u,v)

def BCrigid_slopes(drho,dp,du,dv):   # normal-direction slopes at rigid walls:
    drho[:, 0] = -drho[:, 1]          # rho, p, u are even about the wall -> slopes odd
    drho[:,-1] = -drho[:,-2]
    dp[:, 0]   = -dp[:, 1]
    dp[:,-1]   = -dp[:,-2]
    du[:, 0]   = -du[:, 1]
    du[:,-1]   = -du[:,-2]
    dv[:, 0]   =  dv[:, 1]            # v is odd about the wall -> slope even
    dv[:,-1]   =  dv[:,-2]
    return(drho,dp,du,dv)

def BCperiodic(rho,p,u,v):   # horizontal periodic boundary conditions
    rho[ 0,:]  = rho[-2,:]
    rho[-1,:]  = rho[ 1,:]
    p[ 0,:]    =  p[-2,:]
    p[-1,:]    =  p[ 1,:]
    u[ 0,:]    =  u[-2,:]
    u[-1,:]    =  u[ 1,:]
    v[ 0,:]    =  v[-2,:]
    v[-1,:]    =  v[ 1,:]

    return(rho,p,u,v)

######################################################

def Godunov2(gamma,L,N,c,t_end, IC=IC_kh_springel):
    step   = 0
    W      = L
    M      = N
    
    x,y,dx,dy,rho,p,u,v = IC(L,W,N,M)

    rho,p,u,v       = BCrigid(rho,p,u,v)
    rho,p,u,v       = BCperiodic(rho,p,u,v)
    
    U        = np.zeros((M,N,4))
    U[:,:,0] = rho*dx*dy
    U[:,:,1] = rho*u*dx*dy
    U[:,:,2] = rho*v*dx*dy
    U[:,:,3] = rho*(0.5*(u**2+v**2) + p/(gamma-1)/rho)*dx*dy

    # time
    now   = 0

    # initialize memory
    us   = []
    vs   = []
    ps   = []
    rhos = []
    ts   = []

    us.append(u.copy())
    vs.append(v.copy())
    ps.append(p.copy())
    rhos.append(rho.copy())
    ts.append(now)
    
    cons = [U[1:-1,1:-1][:,:,[0,3]].sum(axis=(0,1))]  #conservation test
    
    # allocate slope arrays
    drho_L = np.zeros((M,N))
    drho_R = np.zeros((M,N))
    drho_U = np.zeros((M,N))
    drho_D = np.zeros((M,N))
    dp_L   = np.zeros((M,N))
    dp_R   = np.zeros((M,N))
    dp_U   = np.zeros((M,N))
    dp_D   = np.zeros((M,N))
    du_L   = np.zeros((M,N))
    du_R   = np.zeros((M,N))
    du_U   = np.zeros((M,N))
    du_D   = np.zeros((M,N))
    dv_L   = np.zeros((M,N))
    dv_R   = np.zeros((M,N))
    dv_U   = np.zeros((M,N))
    dv_D   = np.zeros((M,N))

    Up1    = np.ones((M,N,4))
    Up2    = np.ones((M,N,4))

    while (now<t_end):
        # step 1 - dt
        if (1==0): #(rho<0).any() or (rho==0).any():
            print("bad rho in step # ",step)
        if (p<0).any():
            print("bad p in step # ",step)
        a        = sqrt(gamma*p/rho)
        if(np.isnan(a).any()):
            print('Nan args in a:', np.argwhere(np.isnan(a)))
            sys.exit(0)
        dt       = min(c*(min(dx,dy)/(a+sqrt(u**2+v**2))).min(), t_end-now)
        now     += dt
        step    += 1
        if step%10 == 0:
            print('step ',step, ', time = ',now)
        
        # step 3 - calculate slopes using MinMod
        # horizontal
        drho_L[1:-1,:] = (rho[1:-1,:]-rho[ :-2,:])/dx
        drho_R[1:-1,:] = (rho[2:  ,:]-rho[1:-1,:])/dx
        drho_H= (drho_L*drho_R>0).astype(int) * np.copysign(
            np.minimum(abs(drho_L),abs(drho_R)), drho_L)
        
        dp_L[1:-1,:]   = (p[1:-1,:]  -p[ :-2,:])/dx
        dp_R[1:-1,:]   = (p[2:  ,:]  -p[1:-1,:])/dx
        dp_H  = (dp_L*dp_R>0).astype(int) * np.copysign(
            np.minimum(abs(dp_L),abs(dp_R)), dp_L)

        du_L[1:-1,:]   = (u[1:-1,:]  -u[ :-2,:])/dx
        du_R[1:-1,:]   = (u[2:  ,:]  -u[1:-1,:])/dx
        du_H  = (du_L*du_R>0).astype(int) * np.copysign(
            np.minimum(abs(du_L),abs(du_R)), du_L)

        dv_L[1:-1,:]   = (v[1:-1,:]  -v[ :-2,:])/dx
        dv_R[1:-1,:]   = (v[2:  ,:]  -v[1:-1,:])/dx
        dv_H  = (dv_L*dv_R>0).astype(int) * np.copysign(
            np.minimum(abs(dv_L),abs(dv_R)), dv_L)
        
        # vertical
        drho_U[:,1:-1] = (rho[:,1:-1]-rho[:, :-2])/dy
        drho_D[:,1:-1] = (rho[:,2:  ]-rho[:,1:-1])/dy
        drho_V= (drho_U*drho_D>0).astype(int) * np.copysign(
            np.minimum(abs(drho_U),abs(drho_D)), drho_U)
        
        dp_U[:,1:-1]   = (p[:,1:-1]  -p[:, :-2])/dy
        dp_D[:,1:-1]   = (p[:,2:  ]  -p[:,1:-1])/dy
        dp_V  = (dp_U*dp_D>0).astype(int) * np.copysign(
            np.minimum(abs(dp_U),abs(dp_D)), dp_U)

        du_U[:,1:-1]   = (u[:,1:-1]  -u[:, :-2])/dy
        du_D[:,1:-1]   = (u[:,2:  ]  -u[:,1:-1])/dy
        du_V  = (du_U*du_D>0).astype(int) * np.copysign(
            np.minimum(abs(du_U),abs(du_D)), du_U)

        dv_U[:,1:-1]   = (v[:,1:-1]  -v[:, :-2])/dy
        dv_D[:,1:-1]   = (v[:,2:  ]  -v[:,1:-1])/dy
        dv_V  = (dv_U*dv_D>0).astype(int) * np.copysign(
            np.minimum(abs(dv_U),abs(dv_D)), dv_U)
        
        drho_H, dp_H, du_H, dv_H = BCperiodic(drho_H, dp_H, du_H, dv_H)
        drho_V, dp_V, du_V, dv_V = BCrigid_slopes(drho_V, dp_V, du_V, dv_V)
        
        # steps 4,5 - Solve Riemann problems for quantities on walls
        F      = np.zeros((M-1,N  , 4))
        for k in range(M-1):
            for l in range(N):
                F[k,l,:] =hllc(
                    rho[k,l]+0.5*dx*drho_H[k,l], rho[k+1,l]-0.5*dx*drho_H[k+1,l],
                    p[k,l]  +0.5*dx*dp_H[k,l]  , p[k+1,l]  -0.5*dx*dp_H[k+1,l]  ,
                    u[k,l]  +0.5*dx*du_H[k,l]  , u[k+1,l]  -0.5*dx*du_H[k+1,l]  ,
                    v[k,l]  +0.5*dx*dv_H[k,l]  , v[k+1,l]  -0.5*dx*dv_H[k+1,l]  ,
                    gamma )
                    
        G      = np.zeros((M  ,N-1, 4))                    
        for k in range(M):
            for l in range(N-1):
                G[k,l,:] =hllc(
                    rho[k,l]+0.5*dy*drho_V[k,l], rho[k,l+1]-0.5*dy*drho_V[k,l+1],
                    p[k,l]  +0.5*dy*dp_V[k,l]  , p[k,l+1]  -0.5*dy*dp_V[k,l+1]  ,
                    v[k,l]  +0.5*dy*dv_V[k,l]  , v[k,l+1]  -0.5*dy*dv_V[k,l+1]  ,
                    u[k,l]  +0.5*dy*du_V[k,l]  , u[k,l+1]  -0.5*dy*du_V[k,l+1]  ,
                    gamma )
        switchvar = G[:,:,1].copy()  # hllc assumes calculating flux in u direction,
        G[:,:,1]  = G[:,:,2].copy()  # so we switched the order of hllc arguments, and now we 
        G[:,:,2]  = switchvar.copy() # switch the order of the fluxes
                              # hllc(rho_L,rho_R,p_L,p_R,u_L,u_R,v_L,v_R,gamma)
        
        # No step 6 - no external forces
        # step 7 - first order progression of U
        Up1[1:-1,1:-1,:] = U[1:-1,1:-1,:]                   \
                + dt*dy*( F[ :-1,1:-1,:] - F[1:  ,1:-1,:] )  \
                + dt*dx*( G[1:-1, :-1,:] - G[1:-1,1:  ,:] )

        # step 8 - update primitive variables
        rho = Up1[:,:,0]      /dx/dy
        u   = Up1[:,:,1] /rho /dx/dy
        v   = Up1[:,:,2] /rho /dx/dy
        p   = (gamma-1)*( Up1[:,:,3]/dx/dy - 0.5*rho*(u**2+v**2)) 
        
        # step 9 - new BC
        rho,p,u,v    = BCrigid(rho,p,u,v)
        rho,p,u,v    = BCperiodic(rho,p,u,v)
        
        # step 10 - new slopes
        # horizontal
        drho_L[1:-1,:] = (rho[1:-1,:]-rho[ :-2,:])/dx
        drho_R[1:-1,:] = (rho[2:  ,:]-rho[1:-1,:])/dx
        drho_H= (drho_L*drho_R>0).astype(int) * np.copysign(
            np.minimum(abs(drho_L),abs(drho_R)), drho_L)
        
        dp_L[1:-1,:]   = (p[1:-1,:]  -p[ :-2,:])/dx
        dp_R[1:-1,:]   = (p[2:  ,:]  -p[1:-1,:])/dx
        dp_H  = (dp_L*dp_R>0).astype(int) * np.copysign(
            np.minimum(abs(dp_L),abs(dp_R)), dp_L)

        du_L[1:-1,:]   = (u[1:-1,:]  -u[ :-2,:])/dx
        du_R[1:-1,:]   = (u[2:  ,:]  -u[1:-1,:])/dx
        du_H  = (du_L*du_R>0).astype(int) * np.copysign(
            np.minimum(abs(du_L),abs(du_R)), du_L)

        dv_L[1:-1,:]   = (v[1:-1,:]  -v[ :-2,:])/dx
        dv_R[1:-1,:]   = (v[2:  ,:]  -v[1:-1,:])/dx
        dv_H  = (dv_L*dv_R>0).astype(int) * np.copysign(
            np.minimum(abs(dv_L),abs(dv_R)), dv_L)
        
        # vertical
        drho_U[:,1:-1] = (rho[:,1:-1]-rho[:, :-2])/dy
        drho_D[:,1:-1] = (rho[:,2:  ]-rho[:,1:-1])/dy
        drho_V= (drho_U*drho_D>0).astype(int) * np.copysign(
            np.minimum(abs(drho_U),abs(drho_D)), drho_U)
        
        dp_U[:,1:-1]   = (p[:,1:-1]  -p[:, :-2])/dy
        dp_D[:,1:-1]   = (p[:,2:  ]  -p[:,1:-1])/dy
        dp_V  = (dp_U*dp_D>0).astype(int) * np.copysign(
            np.minimum(abs(dp_U),abs(dp_D)), dp_U)

        du_U[:,1:-1]   = (u[:,1:-1]  -u[:, :-2])/dy
        du_D[:,1:-1]   = (u[:,2:  ]  -u[:,1:-1])/dy
        du_V  = (du_U*du_D>0).astype(int) * np.copysign(
            np.minimum(abs(du_U),abs(du_D)), du_U)

        dv_U[:,1:-1]   = (v[:,1:-1]  -v[:, :-2])/dy
        dv_D[:,1:-1]   = (v[:,2:  ]  -v[:,1:-1])/dy
        dv_V  = (dv_U*dv_D>0).astype(int) * np.copysign(
            np.minimum(abs(dv_U),abs(dv_D)), dv_U)
        
        drho_H, dp_H, du_H, dv_H = BCperiodic(drho_H, dp_H, du_H, dv_H)
        drho_V, dp_V, du_V, dv_V = BCrigid_slopes(drho_V, dp_V, du_V, dv_V)
             
        # steps 11,12 - Solve Riemann problems for quantities on walls
        F      = np.zeros((M-1,N  , 4))
        for k in range(M-1):
            for l in range(N):
                F[k,l,:] =hllc(
                    rho[k,l]+0.5*dx*drho_H[k,l], rho[k+1,l]-0.5*dx*drho_H[k+1,l],
                    p[k,l]  +0.5*dx*dp_H[k,l]  , p[k+1,l]  -0.5*dx*dp_H[k+1,l]  ,
                    u[k,l]  +0.5*dx*du_H[k,l]  , u[k+1,l]  -0.5*dx*du_H[k+1,l]  ,
                    v[k,l]  +0.5*dx*dv_H[k,l]  , v[k+1,l]  -0.5*dx*dv_H[k+1,l]  ,
                    gamma )
                    
        G      = np.zeros((M  ,N-1, 4))                    
        for k in range(M):
            for l in range(N-1):
                G[k,l,:] =hllc(
                    rho[k,l]+0.5*dy*drho_V[k,l], rho[k,l+1]-0.5*dy*drho_V[k,l+1],
                    p[k,l]  +0.5*dy*dp_V[k,l]  , p[k,l+1]  -0.5*dy*dp_V[k,l+1]  ,
                    v[k,l]  +0.5*dy*dv_V[k,l]  , v[k,l+1]  -0.5*dy*dv_V[k,l+1]  ,
                    u[k,l]  +0.5*dy*du_V[k,l]  , u[k,l+1]  -0.5*dy*du_V[k,l+1]  ,
                    gamma )
        switchvar = G[:,:,1].copy()  # hllc assumes calculating flux in u direction,
        G[:,:,1]  = G[:,:,2].copy()  # so we switched the order of hllc arguments, and now we 
        G[:,:,2]  = switchvar.copy() # switch the order of the fluxes
        
        # No step 13 - no external forces
        # step 14 - second order progression of U
        Up2[1:-1,1:-1,:] = Up1[1:-1,1:-1,:]                   \
                + dt*dy*( F[ :-1,1:-1,:] - F[1:  ,1:-1,:] )  \
                + dt*dx*( G[1:-1, :-1,:] - G[1:-1,1:  ,:] )
                
        # step 15 - two-stage RK2 (Toro eq. 15.36): U^{n+1} = (U^n + U~^{n+2})/2
        U   = 0.5*( U + Up2 )
        
        #step 16 - update primitive variables
        rho = U[:,:,0]      /dx/dy
        u   = U[:,:,1] /rho /dx/dy
        v   = U[:,:,2] /rho /dx/dy
        p   = (gamma-1)*( U[:,:,3]/dx/dy - 0.5*rho*(u**2+v**2)) 
        
       # step 2 - boundary conditions (moved here to avoid problems calculating dt)
        rho,p,u,v      = BCrigid(rho,p,u,v)
        rho,p,u,v      = BCperiodic(rho,p,u,v)
        
        
        if step%3 == 0:
            # update memory - only every 3 steps!
            us.append(u.copy())
            ps.append(p.copy())
            rhos.append(rho.copy())
            ts.append(now)
            
            cons.append(U[1:-1,1:-1][:,:,[0,3]].sum(axis=(0,1)))


    rho,p,u,v  = BCrigid(rho,p,u,v)
    
    return( x, y, ts, us, vs, ps, rhos, np.array(cons))

def plotgodunov( x, y, ts, us, vs, ps, rhos):    
    X, Y = np.meshgrid(x,y)
    fig, axs = plt.subplots(4, 4, figsize=(12,12))
    
    l    = len(ts)//16
    indx = 0
    
    for m in range(4):
        for n in range(4):
            if m==3 and n==3:
                indx = -1
            else:
                indx+=l
            axs[m,n].pcolormesh(X[1:-1,1:-1],Y[1:-1,1:-1],np.transpose(rhos[indx][1:-1,1:-1]),shading='gouraud',cmap='turbo') #'RdYlBu_r')
            axs[m,n].set_xlabel('t='+str(round(ts[indx],4)))
    
    fig3, ax3 = plt.subplots(figsize=(7,7))
    ax3.pcolormesh(X[1:-1,1:-1],Y[1:-1,1:-1],np.transpose(rhos[-1][1:-1,1:-1]),shading='gouraud',cmap='turbo')
    plt.show()

def plot_conservation(ts, cons):
    rel = np.abs(cons/cons[0] - 1)
    fig, ax = plt.subplots()
    ax.semilogy(ts[1:], rel[1:,0], label='mass')
    ax.semilogy(ts[1:], rel[1:,1], label='energy')
    ax.set_xlabel('t'); ax.set_ylabel('|Q(t)/Q(0) - 1|'); ax.legend()
    

if __name__ == "__main__":
    x, y, ts, us, vs, ps, rhos, cons = Godunov2(5/3,1,50,0.3,2)
    plot_conservation(ts,cons)
    plotgodunov( x, y, ts, us, vs, ps, rhos)

