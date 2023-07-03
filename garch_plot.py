import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import scipy.stats as st

def run_GARCH(returns_df, sID=50286, test_start=1000, showfirst=600):
    y_actual = returns_df[sID].values
    vn=[]
    hp=[]
    qs=[]
    pps=[]
    am = arch_model(y_actual, vol='GARCH', p=1, o=0, q=1, dist='Normal', mean='Zero')
    end_loc = test_start
    res = am.fit(disp='off', last_obs=end_loc)
    temp = res.forecast(horizon=1, start=end_loc).variance
    temp_ar = np.array(temp.dropna())
    fc_vol = np.reshape(temp_ar, len(temp_ar))
    yt = np.array(y_actual[end_loc:])
    fc_std = np.sqrt(fc_vol)

    am = arch_model(y_actual, vol='EGARCH', p=1, o=0, q=1, dist='Normal', mean='Zero')
    end_loc = test_start
    res = am.fit(disp='off', last_obs=end_loc)
    temp = res.forecast(horizon=1, start=end_loc).variance
    temp_ar = np.array(temp.dropna())
    fc_vol_egarch = np.reshape(temp_ar, len(temp_ar))
    yt = np.array(y_actual[end_loc:])
    fc_std_egarch = np.sqrt(fc_vol_egarch)

    am = arch_model(y_actual, vol='GARCH', p=1, o=1, q=1, dist='Normal', mean='Zero')
    end_loc = test_start
    res = am.fit(disp='off', last_obs=end_loc)
    temp = res.forecast(horizon=1, start=end_loc).variance
    temp_ar = np.array(temp.dropna())
    fc_vol_gjr = np.reshape(temp_ar, len(temp_ar))
    yt = np.array(y_actual[end_loc:])
    fc_std_gjr = np.sqrt(fc_vol_gjr)
    
    #Computing PPS
    yt2 = np.array(y_actual[end_loc:]**2)
    test_size = len(y_actual[end_loc:])
    pps_pre = - (1/test_size) * (-0.5* test_size * np.log(2*np.pi) - 0.5*sum(np.log(fc_vol)) - 0.5*sum(yt2/fc_vol))
    pps.append(pps_pre)
    
    #Predictive Interval
    alpha=0.01
    var_up = 0 - st.norm.ppf(alpha/2)*fc_std #zero mean 99% forecast interval
    var_down = 0 + st.norm.ppf(alpha/2)*fc_std
#    var_up95 = 0 - st.norm.ppf(0.05/2)*fc_std #zero mean 99% forecast interval
#    var_down95 = 0 + st.norm.ppf(0.05/2)*fc_std
    var_up_egarch = 0 - st.norm.ppf(alpha/2)*fc_std_egarch #zero mean 99% forecast interval
    var_down_egarch = 0 + st.norm.ppf(alpha/2)*fc_std_egarch
    var_up_gjr = 0 - st.norm.ppf(alpha/2)*fc_std_gjr #zero mean 99% forecast interval
    var_down_gjr = 0 + st.norm.ppf(alpha/2)*fc_std_gjr

    
    times = pd.to_datetime(returns_df['Dates'].iloc[-test_size:], format='%Y%m%d')

    plt.figure(figsize=(15,7.5))
    plt.plot(times[:showfirst], yt[:showfirst], lw=0.6)
#    plt.plot(times[:showfirst], var_up95[:showfirst], c='r', ls='dashed', lw=0.5, label='95% interval')
#    plt.plot(times[:showfirst], var_down95[:showfirst], c='r', ls='dashed', lw=0.5)
    
    plt.plot(times[:showfirst], var_up_egarch[:showfirst], c='r', marker='x', markevery=5, markersize=4, lw=0.5, label='EGARCH')
    plt.plot(times[:showfirst], var_down_egarch[:showfirst], c='r', marker='x', markevery=5, markersize=4, lw=0.5)

    plt.plot(times[:showfirst], var_up_gjr[:showfirst], c='navy', marker='x', markevery=5, markersize=4, lw=0.5, label='GJR-GARCH')
    plt.plot(times[:showfirst], var_down_gjr[:showfirst], c='navy', marker='x', markevery=5, markersize=4, lw=0.5)

    plt.plot(times[:showfirst], var_up[:showfirst], c='lime', marker='x', markevery=5, markersize=4, lw=0.5, label='GARCH')
    plt.plot(times[:showfirst], var_down[:showfirst], c='lime', marker='x', markevery=5, markersize=4, lw=0.5)
    plt.legend()
    plt.grid(True, 'major', ls='--', lw=.5, c='k', alpha=.15)
    plt.grid(True, 'minor', ls=':', lw=.5, c='k', alpha=.3)
    plt.title('99% One-Step-Ahead Forecast Intervals',size=15)
    plt.tight_layout()
    plt.savefig('/Users/zt237/Dropbox/Research_Thoughts/var_garch/gph/var.pdf')

    
    #violation number
    count_up=0
    count_down=0
    for i in range(len(yt)):
        if yt[i]<=var_down[i]:
            count_down += 1
        else:
            count_down += 0
        
        if yt[i]>=var_up[i]:
            count_up += 1
        else:
            count_up += 0
    
    violation_num = count_up + count_down
    vn.append(violation_num)
    
    #VaR
    VaR = st.norm.ppf(alpha)*fc_std
    
    #hit percentage
    count_VaR=0
    indicator=[]
    for i in range(len(yt)):
        if yt[i]<=VaR[i]:
            count_VaR += 1
            indicator.append(1)
        else:
            count_VaR += 0
            indicator.append(0)
            
    hit_percent = count_VaR/len(yt)
    hp.append(hit_percent)
    
    #QS
    diff = np.array([yt - VaR]).T
#    alpha = 0.01
    diff_ind = alpha - np.array([indicator])
    qs_pre = (diff_ind @ diff)/len(yt)
    qs.append(float(qs_pre))

    d = {'PPS': pps, 'Violation Number': vn, 'QS': qs, 'Hit Percentage': hp}
    df = pd.DataFrame(data=d)
    return df


returns_df = pd.read_excel('/Users/zt237/Dropbox/Research_Thoughts/var_garch/src/Returns_Clean.xlsx')
flows_df = pd.read_excel('/Users/zt237/Dropbox/Research_Thoughts/var_garch/src/Flows_Clean.xlsx')

#data = np.array([returns_df[50286].values]).T

run_GARCH(returns_df, sID=50286, test_start=1100, showfirst=750)









