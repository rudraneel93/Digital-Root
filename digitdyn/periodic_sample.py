import random, math, sys
def s(n):
    t=0
    while n: t+=n%10; n//=10
    return t
def f(n): return n+s(n) if n%2 else n-s(n)
random.seed(11)
for D in [int(v) for v in sys.argv[1].split(',')]:
    n_s=int(sys.argv[2]); per=0; two=0; lens={}; censored=0
    for _ in range(n_s):
        n=random.randrange(10**(D-1),10**D)
        seen={n:0}; x=n; k=0
        while True:
            x=f(x); k+=1
            if x in seen: break
            seen[x]=k
            if k>20000: censored+=1; break
        if x==n:
            per+=1; lens[k]=lens.get(k,0)+1
            if k==2: two+=1
    p=per/n_s; se=math.sqrt(p*(1-p)/n_s)
    print(f"D={D:4d}: periodic {p:.4f} +- {se:.4f}; 2-cycles {two/n_s:.4f}; "
          f"sqrt(log10 D)*periodic {p*math.sqrt(math.log10(D)):.4f}; censored at 20000 steps: {censored}; lengths {dict(sorted(lens.items()))}", flush=True)
