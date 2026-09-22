from dataclasses import asdict,dataclass
import platform,numpy as np,scipy
import re
from inverse_em.errors import SchemaValidationError
from inverse_em.physics.vendor import UPSTREAM_COMMIT,UPSTREAM_REPOSITORY,UPSTREAM_SOURCE_PATH,UPSTREAM_SOURCE_SHA256
from .canonical import canonical_sha256

def _sha256(value,name):
    if not isinstance(value,str) or not re.fullmatch(r"[0-9a-f]{64}",value):raise SchemaValidationError(f"{name} must be 64 lowercase hexadecimal characters")

@dataclass(frozen=True)
class PhysicsPin:
    upstream_repository:str=UPSTREAM_REPOSITORY;upstream_commit:str=UPSTREAM_COMMIT;upstream_source_path:str=UPSTREAM_SOURCE_PATH;upstream_source_sha256:str=UPSTREAM_SOURCE_SHA256;license:str="MIT License; Copyright (c) 2026 Thomas D. Papadopoulos"
    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}",self.upstream_commit):raise SchemaValidationError("upstream_commit must be lowercase hexadecimal Git identity")
        _sha256(self.upstream_source_sha256,"upstream_source_sha256")
        if not self.upstream_repository.startswith("https://") or not self.upstream_source_path.strip():raise SchemaValidationError("Invalid upstream source identity")
@dataclass(frozen=True)
class ReferenceFixtureProvenance:
    fixture_name:str;fixture_sha256:str;configuration_sha256:str;source_positions_sha256:str;observation_grid_sha256:str;dtype:str;python:str;numpy:str;scipy:str;physics_pin:PhysicsPin=PhysicsPin()
    def __post_init__(self):
        if not self.fixture_name.strip() or self.dtype!="complex128":raise SchemaValidationError("Invalid reference fixture identity")
        for name in ("fixture_sha256","configuration_sha256","source_positions_sha256","observation_grid_sha256"):_sha256(getattr(self,name),name)
    @classmethod
    def create(cls,name,fixture,config,sources,grid,dtype="complex128"):return cls(name,fixture,config,sources,grid,dtype,platform.python_version(),np.__version__,scipy.__version__)
    @property
    def sha256(self):return canonical_sha256(asdict(self))
