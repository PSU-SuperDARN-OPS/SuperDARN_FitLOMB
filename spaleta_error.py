# code from jef for generating errors

def phase_fit_error(signal,lag_time_secs,tfreq_hz,v_estimate):
    import numpy as N    
    C = 299792458.   # m/s
    Yphase=N.angle(signal)
    v=v_estimate # This could be either mean or peak whatever is being reported
    ssx=0.
    sse=0.
    if len(Yphase) > 2:
        ns=1.
        bayes_signal=N.exp(1J*2 * N.pi * 2*tfreq_hz*ns*v/C* lag_time_secs)
        Bphase=N.angle(bayes_signal)

        # build the phase residuals
        for i in xrange(len(Yphase)):
            res=Bphase[i]-Yphase[i] 
            # Assume fit is at most phone phase wrap away at a particular lag.
            resY=N.min([N.abs(res+2*N.pi),N.abs(res-2*N.pi),N.abs(res)])
            sse+=resY**2

        # ssx calculated for fit forced through zero phase at time=0
        ssx=N.sum(lag_time_secs**2)

        # Calculate Errors associated with the fit

        # Standard error
        se=N.sqrt(sse/(len(Yphase)-1))  # Units of radian
        phi_sigma=se
        # Slope error with fit forced through intercept of 0
        slope_sigma=se/N.sqrt(ssx) # Units of rad/sec
        # Rescaled slope error in terms of velocity
        v_sigma=(C*slope_sigma)/(4*N.pi*tfreq_hz)  # Units of m/s
    else:
        se=N.nan
        phi_sigma=N.nan
        slope_sigma=N.nan
        v_sigma=N.nan
    return phi_sigma,slope_sigma,v_sigma
