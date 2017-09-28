#! /usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
# import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smformula
import cartopy.crs as ccrs
import pymc3 as pm


coretop_path = './data_parsed/coretops-2017-09-26.csv'


coretops_raw = pd.read_csv(coretop_path)
coretops_raw.sort_values('species', inplace=True)
coretops = coretops_raw.query('temp_ann > 0').copy()

# sns.lmplot(y='d18oc', x='temp_ann', data=coretops)
# plt.show()


## MCMC
n_species = len(coretops['species'].unique())
idx_species = pd.Categorical(coretops['species']).codes

with pm.Model() as model0:

    # Hyperparameters
    mu_a = pm.Normal('mu_alpha', mu=0, sd=1)
    sigma_a = pm.HalfCauchy('sigma_alpha', beta=1)

    mu_b = pm.Normal('mu_beta', mu=0, sd=1)
    sigma_b = pm.HalfCauchy('sigma_beta', beta=1)

    # Intercept and slope
    a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=n_species)
    b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=n_species)
    # Model error
    # eps = pm.Uniform('eps', lower=0, upper=1, shape=n_species)
    eps = pm.HalfCauchy('eps', beta=1, shape=n_species)

    # Likelihood
    # d18oc_est = a[idx_species] + b[idx_species] * coretops['temp_ann'] + (coretops['d18osw'] - 0.27)
    d18oc_est = a[idx_species] + b[idx_species] * coretops['temp_ann']
    likelihood = pm.Normal('y', mu=d18oc_est, sd=eps[idx_species], 
                           observed=coretops['d18oc'])

    trace0 = pm.sample(5000, njobs=1)


pm.summary(trace0)

## Need multiple jobs for this:
# target_stat = 'mu_alpha'
# print('Rhat({}) = {}'.format(target_stat, 
#     pm.diagnostics.gelman_rubin(trace0)[target_stat]))


pm.traceplot(trace0[100:], priors=[mu_a.distribution, mu_b.distribution,
                                   None, None,
                                   sigma_a.distribution, sigma_b.distribution,
                                   eps.distribution]);plt.tight_layout();plt.show()

pm.forestplot(trace0)


ppc = pm.sample_ppc(trace0, samples=500, model=model)
np.asarray(ppc['y']).shape

ax = plt.subplot()
sns.distplot([y.mean() for y in ppc['y']], kde=False, ax=ax)
ax.axvline(coretops['d18oc'].mean())
ax.set(title='Posterior predictive of the mean', 
       xlabel='mean(d18oc)', ylabel='Frequency');


# Plot the simulations of fits for each spp.
synth_temp = np.arange(0, 31)
for i in np.unique(idx_species):
    spp = coretops.species.unique()[i]
    ax = plt.subplot(2, 3, i+1)
    spp_msk = coretops.species.unique() == spp
    y = trace0['alpha'][:, spp_msk] + trace0['beta'][:, spp_msk] * synth_temp
    ax.plot(np.broadcast_to(synth_temp, (y.shape[0], len(synth_temp))).T, 
            y.T, alpha = 0.01, label=spp, 
            color=list(matplotlib.colors.TABLEAU_COLORS.values())[i])
    coretops.query("species == '{}'".format(spp)).plot(y='d18oc', x='temp_ann', 
        ax=ax, kind='scatter', color='black', alpha=0.5, zorder=10)
    ax.set_xlim(0, 30)
    ax.set_ylim(-4, 4.5)
    ax.set_title(spp)
    ax.grid(True)
plt.tight_layout()
plt.show()


# Plot residuals of the median model for each spp.
for i in np.unique(idx_species):
    spp = coretops.species.unique()[i]
    coretops_sub = coretops.query("species == '{}'".format(spp))
    ax = plt.subplot(2, 3, i+1)
    spp_msk = coretops.species.unique() == spp
    alpha = np.median(trace0['alpha'][:, spp_msk])
    beta = np.median(trace0['beta'][:, spp_msk])
    y = alpha + beta * coretops_sub['temp_ann']
    resids = coretops_sub['d18oc'] - y
    resids_smoothed = sm.nonparametric.lowess(resids, coretops_sub['temp_ann'], frac=0.75)
    ax.plot(resids_smoothed[:, 0], resids_smoothed[:, 1], label='LOWESS', linestyle=':', color='black')

    ax.scatter(x=coretops_sub['temp_ann'], y=resids, marker='.', label=spp, 
               color=list(matplotlib.colors.TABLEAU_COLORS.values())[i])
    ax.axhline(y=0, color='black')
    ax.set_title(spp)
    ax.set_ylabel('Obs. - pred. (d18Oc)')
    ax.set_xlabel('Annual SST (C)')
    ax.grid(True)
plt.tight_layout()
plt.show()



# with pm.Model() as model:
    # pm.glm.GLM.from_formula('d18oc ~ temp_ann', coretops)
    # trace=pm.sample(3000, njobs=2)
# pm.traceplot(trace[100:]);plt.tight_layout()


# with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
#     # Define priors
#     sigma = pm.HalfCauchy('sigma', beta=10, testval=1.0)
#     intercept = pm.Normal('Intercept', 0, sd=20)
#     x_coeff = pm.Normal('x', 0, sd=20)
#     # Define likelihood
#     likelihood = pm.Normal('y', mu=intercept + x_coeff * coretops['temp_ann'],
#                            sd=sigma, observed=coretops['d18oc'])
#     # Inference!
#     trace = pm.sample(5000, njobs=1) # draw 5000 posterior samples using NUTS sampling

