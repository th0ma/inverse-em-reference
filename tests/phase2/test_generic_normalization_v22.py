import numpy as np
import pytest
from inverse_em.config import InverseTask
from inverse_em.errors import SchemaValidationError
from inverse_em.normalization import FrozenChannelNormalizer,GenericChannelStatsAccumulator
def data():return np.arange(10*4*30,dtype=float).reshape(10,4,30)/17+np.arange(4)[None,:,None]
def test_generic_one_shot_chunk_merge_and_transform_reference():
    x=data();one=GenericChannelStatsAccumulator().update(x).freeze();merged=GenericChannelStatsAccumulator().update(x[:1]).merge(GenericChannelStatsAccumulator().update(x[1:])).freeze();mean=x.mean((0,2));std=x.std((0,2),ddof=0)
    for n in (one,merged):assert np.allclose(n.mean,mean,atol=2e-14) and np.allclose(n.std,std,atol=2e-14) and (n.observation_count,n.scalar_count_per_channel)==(10,300)
    frozen=FrozenChannelNormalizer(InverseTask.S1,mean,std,10,300);before=x.copy();assert np.allclose(frozen.transform(x),(x-mean[None,:,None])/std[None,:,None]) and np.array_equal(x,before)
def test_frozen_task_and_immutability():
    for task in ("s1",np.str_("s1"),object(),True,1):
        with pytest.raises(SchemaValidationError):FrozenChannelNormalizer(task,np.zeros(4),np.ones(4),1,30)
    n=FrozenChannelNormalizer(InverseTask.S1,np.zeros(4),np.ones(4),1,30)
    for x in (n.mean,n.std):
        with pytest.raises(ValueError):x.setflags(write=True)
