# v0.0.1a5

## Bug fixes

## Enhancements


# v0.0.1a4

## Bug fixes

* Fix bad keyword argument in `predict_d18oc()`.


# v0.0.1a3

## Enhancements

* `predict_d18oc` now has `seasonal_seatemp arg.

# v0.0.1a2

## Enhancements

* Update SST statistics for forams.

* Compatibility for new and legacy species names. These name changes will be automatic within functions: 
  'N. pachyderma sinistral' now becomes 'N. pachyderma'. All ruber sub species now become 
  'G. ruber'. 'G. sacculifer' now automatically is switched to 'T. sacculifer'. This means that 
  `bayfox` >v0.0.1a2 will need to be installed.
  
* Removed `predict_seatemp()` as should just use `bayfox` for this.


# v0.0.1a1

* New function `foram_sst_minmax(foram)` to get min and max SSTs observed in the 
    coretop record for a given foram.

* Now use `bayfox` MCMC parameters for d18Oc and seatemp predictions.

* Docstring cleanup and fluffing.


# v0.0.1a0

* Initial release.
