# v0.0.1a3

## Enhancements


## Bug fixes


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
