"""Synthetic predictor-only token-grid cost; does not update saved policy weights."""
import copy
import json
from pathlib import Path
import time

import torch
from tactile_refinement.representation import Predictor


class GridPredictor(Predictor):
    """Diagnostic-only generalization of the production 32-token reshape."""
    def forward(self,current,context,context_mask):
        ordered=current+(self.sensor+self.position).reshape(1,current.shape[1],-1)
        memory=torch.cat((ordered,self.prefix_adapter(context)),1)
        padding=torch.cat((torch.zeros(current.shape[:2],dtype=torch.bool,device=current.device),
                           ~context_mask.bool()),1)
        x=self.queries.expand(current.shape[0],-1,-1)
        for block in self.blocks:x=block(x,memory,memory_key_padding_mask=padding)
        return self.norm(x)


def main():
    torch.set_num_threads(4)
    checkpoint=torch.load('runs/tactile_refinement/encoder/final.pt',weights_only=True,map_location='cpu',mmap=True)
    width,context_width=checkpoint['model']['predictor.prefix_adapter.weight'].shape
    del checkpoint
    results=[]
    for positions in (16,300):
        torch.manual_seed(42)
        model=GridPredictor(context_width,width)
        model.position=torch.nn.Parameter(torch.randn(1,positions,width)*.02)
        model.queries=torch.nn.Parameter(torch.randn(1,2*positions,width)*.02)
        model=model.cuda().train()
        current=torch.randn(16,2*positions,width,device='cuda')
        context=torch.randn(16,512,context_width,device='cuda')
        mask=torch.ones(16,512,dtype=torch.bool,device='cuda')
        target=torch.randn_like(current)
        times=[];extra=[]
        for iteration in range(7):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(current,context,mask)
                loss=(pred-target).square().mean()
            loss.backward();torch.cuda.synchronize()
            if iteration>=2:
                times.append(time.perf_counter()-start)
                extra.append((torch.cuda.max_memory_allocated()-baseline)/2**20)
        results.append({'tokens_per_finger':positions,'total_tactile_tokens':2*positions,
                        'batch_size':16,'context_tokens':512,'context_width':context_width,
                        'mean_forward_backward_seconds':sum(times)/len(times),
                        'max_additional_allocated_mib':max(extra),
                        'bf16_tactile_tokens_mib_per_example':2*positions*width*2/2**20})
        del model,current,context,mask,target,pred,loss;torch.cuda.empty_cache()
    output=Path('runs/diagnostics/encoder_pooling_probe/predictor_token_cost.json')
    output.write_text(json.dumps({'scope':'synthetic predictor-only forward/backward, 16 examples and 512 prefix tokens; excludes baseline, expert, optimizer and dataset',
                                  'measurements':results},indent=2)+'\n')
    print(output.read_text())


if __name__=='__main__':main()
