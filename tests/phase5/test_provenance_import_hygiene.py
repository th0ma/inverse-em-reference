import os
from pathlib import Path
import subprocess
import sys

import torch

from inverse_em.config.localization import LocalizationScientificConfig
from inverse_em.provenance.localization import LocalizationInfrastructureReceipt


def test_descriptive_receipt_is_canonical_and_not_authority():
    config=LocalizationScientificConfig()
    receipt=LocalizationInfrastructureReceipt(config.sha256,"phase5-common-localizer-v1",
        config.surrogates.electric_checkpoint_sha256,config.surrogates.magnetic_checkpoint_sha256,str(torch.__version__))
    assert len(receipt.sha256)==64 and receipt.sha256==LocalizationInfrastructureReceipt(**receipt.__dict__).sha256
    assert not hasattr(receipt,"authorize") and not hasattr(receipt,"capability")


def test_import_hygiene(tmp_path):
    code="""import pathlib,torch,numpy as np
root=pathlib.Path.cwd();before={str(p.relative_to(root)) for p in root.rglob('*')};ts=torch.get_rng_state().clone();ns=np.random.get_state()
import inverse_em.localization,inverse_em.config.localization,inverse_em.provenance.localization
assert torch.equal(ts,torch.get_rng_state());now=np.random.get_state();assert ns[0]==now[0] and np.array_equal(ns[1],now[1]);assert before=={str(p.relative_to(root)) for p in root.rglob('*')}
"""
    environment=dict(os.environ);environment["PYTHONPATH"]=str(Path(__file__).resolve().parents[2]/"src");environment["PYTHONDONTWRITEBYTECODE"]="1"
    subprocess.run([sys.executable,"-B","-c",code],cwd=tmp_path,env=environment,check=True)
