"""Extract only deformation tensors from the pinned public PyTorch ZIP via ranges.

Metadata uses PyTorch's restricted weights-only unpickler and meta storages.
No arbitrary pickle globals or upstream model code are executed.
"""
import io
import json
from pathlib import Path
import zipfile

import requests
import torch
from torch._weights_only_unpickler import Unpickler
from huggingface_hub import HfApi

from tactile_refinement.representation import DeformEncoder

REPO='miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6'
REVISION='62efb3bcb45a3df0e088c8909d759b582cfb98af'
OUTPUT=Path('runs/diagnostics/encoder_pooling_probe/trex_deform_encoder.pt')


class RemoteZip(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size,self.pos=url,size,0
        self.session=requests.Session(); self.cache={}; self.bytes_read=0

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset+(0 if whence==0 else self.pos if whence==1 else self.size)
        if self.pos<0:raise ValueError('negative seek')
        return self.pos

    def read(self,size=-1):
        stop=min(self.size,self.pos+size) if size>=0 else self.size
        if stop-self.pos>16*1024*1024:raise ValueError('Unexpected bulk read')
        parts=[]
        while self.pos<stop:
            start=(self.pos//(1024*1024))*(1024*1024)
            end=min(start+1024*1024,self.size)-1
            if start not in self.cache:
                r=self.session.get(self.url+f'?range_block={start}',headers={'Range':f'bytes={start}-{end}'},timeout=60,stream=True)
                if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                    r.close();raise ValueError('Server did not honor exact byte range')
                self.cache[start]=r.content;self.bytes_read+=len(r.content);r.close()
            take=min(stop-self.pos,end-self.pos+1)
            parts.append(self.cache[start][self.pos-start:self.pos-start+take]);self.pos+=take
        return b''.join(parts)


class MetadataUnpickler(Unpickler):
    def __init__(self,stream):
        super().__init__(stream);self.storages={};self.keys={}

    def persistent_load(self,pid):
        typename,storage_type,key,location,numel=pid
        if typename not in ('storage',b'storage'):raise ValueError('Unexpected persistent type')
        dtype=storage_type.dtype
        if key not in self.storages:
            size=numel*torch.empty((),dtype=dtype).element_size()
            raw=torch.UntypedStorage(size,device='meta')
            self.storages[key]=torch.storage.TypedStorage(wrap_storage=raw,dtype=dtype,_internal=True)
            self.keys[raw._cdata]=key
        return self.storages[key]


def main():
    info=HfApi().model_info(REPO,revision=REVISION,files_metadata=True)
    item=next(x for x in info.siblings if x.rfilename=='model.pt')
    remote=RemoteZip(f'https://huggingface.co/{REPO}/resolve/{REVISION}/model.pt',item.size)
    with zipfile.ZipFile(remote) as archive:
        name=next(x for x in archive.namelist() if x.endswith('/data.pkl'))
        prefix=name.removesuffix('data.pkl')
        if prefix+'byteorder' in archive.namelist():assert archive.read(prefix+'byteorder')==b'little'
        parser=MetadataUnpickler(io.BytesIO(archive.read(name)))
        metadata=parser.load()
        state={}
        for key,tensor in metadata.items():
            if not key.startswith('deform_encoder.'):continue
            storage_key=parser.keys[tensor.untyped_storage()._cdata]
            raw=bytearray(archive.read(prefix+'data/'+storage_key))
            value=torch.frombuffer(raw,dtype=tensor.dtype).as_strided(tensor.size(),tensor.stride(),tensor.storage_offset()).clone()
            state[key.removeprefix('deform_encoder.')]=value
    model=DeformEncoder();model.load_state_dict(state,strict=True)
    assert all(torch.isfinite(x).all() for x in state.values())
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'encoder':state,'repo':REPO,'revision':REVISION,'source_lfs_sha256':item.lfs.sha256,
                'range_download_bytes':remote.bytes_read},OUTPUT)
    print(json.dumps({'output':str(OUTPUT),'keys':len(state),'downloaded_bytes':remote.bytes_read,'full_file_bytes':item.size}),flush=True)


if __name__=='__main__':main()
