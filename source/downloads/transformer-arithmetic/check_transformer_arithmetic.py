"""Recompute the blog's examples without loading model weights.

Scope: bias-free decoder, two RMSNorms/block, SwiGLU, no learned positions,
untied embedding/head. Mixtral uses all-MoE blocks with a bias-free router.
The shape ledger follows X @ W (PyTorch stores Linear.weight transposed).
"""
from dataclasses import dataclass, replace
from math import prod


@dataclass(frozen=True)
class Model:
    name: str
    d: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn_dim: int
    vocab: int
    experts: int = 1
    top_k: int = 1

    def block_shapes(self, active=False):
        d, q, kv, f = self.d, self.heads*self.head_dim, self.kv_heads*self.head_dim, self.ffn_dim
        shapes = {'q':(d,q), 'k':(d,kv), 'v':(d,kv), 'o':(q,d),
                  'attention_norm':(d,), 'ffn_norm':(d,)}
        n = self.top_k if active else self.experts
        for i in range(n):
            for name, shape in [('gate',(d,f)),('up',(d,f)),('down',(f,d))]:
                shapes[f'expert_{i}.{name}'] = shape
        if self.experts > 1:
            shapes['router'] = (d,self.experts)
        return shapes

    def block_count(self, active=False):
        return sum(prod(shape) for shape in self.block_shapes(active).values())

    def total(self, active=False):
        return self.layers*self.block_count(active) + 2*self.vocab*self.d + self.d

    def linear(self, active=False):
        return sum(prod(shape) for shape in self.block_shapes(active).values() if len(shape)==2)

    def cache(self, batch, length, bytes_per_element=2):
        # Count K and V independently from their full tensor shapes.
        one_tensor = (self.layers,batch,self.kv_heads,length,self.head_dim)
        return (prod(one_tensor)+prod(one_tensor))*bytes_per_element


def main():
    llama2 = Model('Llama-2 7B',4096,32,32,32,128,11008,32000)
    llama3 = Model('Llama-3 8B',4096,32,32,8,128,14336,128256)
    mixtral = Model('Mixtral 8x7B',4096,32,32,8,128,14336,32000,8,2)
    assert llama2.total() == 6_738_415_616
    assert llama3.total() == 8_030_261_248
    assert mixtral.total() == 46_702_792_704
    assert mixtral.total(active=True) == 12_879_925_248
    assert llama2.block_count() == 202_383_360
    assert llama3.block_count() == 218_112_000
    assert mixtral.block_count() == 1_451_270_144
    for m in (llama2,llama3,mixtral):
        # Independently check the closed form against individual matrix shapes.
        d, a, kv, f = m.d,m.heads*m.head_dim,m.kv_heads*m.head_dim,m.ffn_dim
        formula = m.layers*(2*d*a+2*d*kv+3*m.experts*d*f+2*d+(d*m.experts if m.experts>1 else 0))+2*m.vocab*d+d
        assert m.total() == formula
        print(f'{m.name:16} total={m.total():,} active={m.total(True):,}')

    assert 8*8 + 2*8*4 + 8*8 == 192
    assert llama3.cache(1,8192) == 2**30
    assert llama2.cache(1,4096) == 2*2**30
    assert llama3.cache(1,1) == 128*1024
    assert llama2.cache(1,1) == 512*1024
    assert 32*2*8*(4096+4096) == 4_194_304
    assert replace(llama3,vocab=138256).total()-llama3.total() == 81_920_000
    assert replace(llama3,kv_heads=4).total()-llama3.total() == -134_217_728
    assert replace(llama3,ffn_dim=14592).total()-llama3.total() == 100_663_296
    available = (24-3)*2**30 - 2*llama3.total()
    assert available//llama3.cache(1,8192) == 6
    assert available//llama3.cache(1,1) == 49_499

    # Sequence arithmetic: first output comes from prefill, not another decode.
    s, g, b = 1024,128,1
    r = g-1
    l,d,v,p = llama3.layers,llama3.d,llama3.vocab,llama3.linear()
    assert p == 218_103_808
    linear = 2*b*l*p*(s+r)
    core = 2*b*l*d*s*(s+1) + 4*b*l*d*(r*s+r*(r+1)//2)
    head = 2*b*g*d*v
    explicit = 2*b*l*p*s + 2*b*l*d*s*(s+1) + 2*b*d*v
    for j in range(1,r+1):
        explicit += 2*b*l*p + 4*b*l*d*(s+j) + 2*b*d*v
    assert linear+core+head == explicit == 16_548_475_437_056
    assert llama3.cache(1,s+r)/2**20 == 143.875
    assert core == 347_590_361_088
    assert head == 134_486_163_456
    print(f'Generation FLOPs: {explicit:,}; final KV: {llama3.cache(1,s+r)/2**20} MiB')
    decode_flops = 2*(l*p+d*v) + 4*l*8192*d
    traffic = 2*(l*p+d*v) + llama3.cache(1,8192)
    assert decode_flops == 19_304_284_160
    print(f'Hypothetical compute lower bound: {decode_flops/200e12*1000:.4f} ms')
    print(f'Hypothetical bandwidth lower bound: {traffic/1e12*1000:.4f} ms')
    # General attention width must not silently assume h*d_h == d.
    unusual = Model('nonstandard head_dim',8,1,4,2,3,12,20)
    assert sum(prod(unusual.block_shapes()[k]) for k in ('q','k','v','o')) == 288
    print('All arithmetic assertions passed.')


if __name__ == '__main__':
    main()
