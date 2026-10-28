# -*- coding: utf-8 -*-
"""
Evolutionary modelers — statistical models fitted to each component's difference data.
"""

from .._registry import register_modeler
from ._gmm_bic import gmm_bic_model_1d
from ._gmm_bic_post import gmm_bic_model_post

register_modeler("gmm_bic_model_1d", gmm_bic_model_1d)
register_modeler("gmm_bic_model_post", gmm_bic_model_post)

from ._robust import RobustDeviationModel
register_modeler("robust_deviation", RobustDeviationModel)
