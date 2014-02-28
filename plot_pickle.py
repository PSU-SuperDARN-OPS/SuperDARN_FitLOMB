from rawacf_to_fitlomb import *
import pdb
import cPickle as pickle


# assumes times are the same between rawacf and fitacf
def compare_lombfit(fitacf, fitlombs):
    fitlombs = pickle.load(open(data, 'rb'))
    fitacfs = DMapFile(files=[fitacf]) 

    w_fitacf = []
    w_lomb = [] 

    v_fitacf = [] 
    v_lomb = []

    p_fitacf = []
    p_lomb = []

    for fitlomb in fitlombs:
        fitacf = fitacfs[fitlomb.recordtime]
        slist = fitacf['slist']

        w_fitacf.append(np.array(fitacf['w_l']))
        w_lomb.append(np.array([fitlomb.w_l[:,0][s] for s in slist]))

        v_fitacf.append(np.array(fitacf['v']))
        v_lomb.append(np.array([fitlomb.v_l[:,0][s] for s in slist]))
        
        # NOTE: fitacf uses log power?.. scale 10 * log10
        p_fitacf.append(np.array(fitacf['p_l']))
        p_lomb.append(10 * np.log10(np.array([fitlomb.p_l[:,0][s] for s in slist])))
    
        for (i,s) in enumerate(slist):
            fit = fitlomb.lfits[s][0]
            # plot fitacf and fitlomb fits
            samples = fit['samples']
            fitsignal = fit['signal']
            t = fit['t']
            noise = fitacf['noise.sky']
            amp = noise * (10 ** (fitacf['p_l'][i]/10))
            f = (fitacf['v'][i] * 2 * fitlomb.tfreq * 1e3) / C
            alf = fitacf['w_l'][i] # * ??? 
            facf = amp * np.exp(1j * 2 * np.pi * f * t) * np.exp(-alf * t)
            print f
            print fit['frequency']
            print alf
            print fit['alpha']
            plt.plot(samples)
            plt.plot(fitsignal)
            plt.plot(facf)

            plt.legend(['samples', 'fitlomb', 'fitacf'])
            plt.show()

if __name__ == "__main__":
    data = 'beam9_mcm.p'
    fitacf = '20130320.0801.00.mcm.a.fitacf'
    fitlombs = pickle.load(open(data, 'rb'))

    compare_lombfit(fitacf, fitlombs)
    
#    for lomb in fitlombs:
#
#        lomb.qwle_thresh = 70
#        lomb.qvle_thresh = 70
#        lomb.ProcessPeaks()
#        PlotMixed(lomb)

    PlotVRTI(fitlombs, 9)
    PlotWRTI(fitlombs, 9)

       
